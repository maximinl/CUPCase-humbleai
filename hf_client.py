"""
Standalone HuggingFace client for local model inference.
Adapted from HuggingFaceClient — no external deps beyond transformers/torch.
Provides an OpenAI-compatible async interface for drop-in pipeline use.
"""

import asyncio
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Lightweight response objects mimicking OpenAI's API
# ---------------------------------------------------------------------------

@dataclass
class _Message:
    content: str
    role: str = "assistant"

@dataclass
class _Choice:
    message: _Message
    index: int = 0
    finish_reason: str = "stop"

@dataclass
class _ChatCompletion:
    choices: list
    model: str = ""
    usage: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# OpenAI-compatible async wrapper
# ---------------------------------------------------------------------------

class _ChatCompletions:
    """Mimics openai_client.chat.completions.create()"""

    def __init__(self, hf_client: "HFClient"):
        self._client = hf_client

    async def create(self, model: str = "", messages: list = None,
                     temperature: float = 0, response_format: dict = None,
                     **kwargs) -> _ChatCompletion:
        old_temp = self._client.temperature
        self._client.temperature = temperature
        try:
            content = await self._client.acompletion(messages)
        finally:
            self._client.temperature = old_temp

        return _ChatCompletion(
            choices=[_Choice(message=_Message(content=content))],
            model=self._client.model_name,
            usage={
                "prompt_tokens": self._client.last_prompt_tokens,
                "completion_tokens": self._client.last_completion_tokens,
            },
        )

class _Chat:
    def __init__(self, hf_client: "HFClient"):
        self.completions = _ChatCompletions(hf_client)


# ---------------------------------------------------------------------------
# Core HuggingFace Client
# ---------------------------------------------------------------------------

class HFClient:
    """
    Local HuggingFace model that exposes an AsyncOpenAI-compatible interface.

    Usage:
        client = HFClient("meta-llama/Llama-3.3-70B-Instruct", quantize="4bit")
        # Use exactly like AsyncOpenAI:
        res = await client.chat.completions.create(
            model="ignored", messages=[...], temperature=0
        )
        content = res.choices[0].message.content
    """

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        device: str = "auto",
        quantize: Optional[str] = None,
        num_gpus: Optional[int] = None,
        trust_remote_code: bool = True,
        truncate_input_tokens: int = 4096,
        **kwargs,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.device_pref = device
        self.quantize = quantize
        self.num_gpus = num_gpus
        self.trust_remote_code = trust_remote_code
        self.truncate_input_tokens = truncate_input_tokens
        self.enable_thinking = kwargs.pop("enable_thinking", False)

        if self.quantize not in (None, "4bit", "8bit"):
            raise ValueError("quantize must be one of None, '4bit', or '8bit'")

        # Usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

        self._model = None
        self._tokenizer = None
        self._device_str = "cpu"
        self._load()

        # OpenAI-compatible interface
        self.chat = _Chat(self)

    # ---- Device / dtype helpers ----

    def _choose_device(self) -> str:
        import torch
        if self.device_pref == "cpu":
            return "cpu"
        if (self.device_pref in ("auto", "cuda")) and torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def _infer_dtype(self, device: str):
        import torch
        if device.startswith("cuda"):
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    # ---- Model loading ----

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self._choose_device()
        torch_dtype = self._infer_dtype(device)
        self._device_str = device

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

        print(f"[HFClient] Loading {self.model_name} on {device} (dtype={torch_dtype})")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            token=hf_token,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        num_gpus_to_use = min(self.num_gpus, available_gpus) if self.num_gpus else available_gpus

        if num_gpus_to_use > 1:
            device_map = "auto"
            print(f"[HFClient] Using {num_gpus_to_use} GPUs with device_map='auto'")
        else:
            device_map = "auto" if device.startswith("cuda") else None

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "token": hf_token,
        }

        if self.quantize == "8bit":
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["device_map"] = "auto"
            print("[HFClient] Using 8-bit quantization")
        elif self.quantize == "4bit":
            from transformers import BitsAndBytesConfig
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch_dtype
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["device_map"] = "auto"
            print("[HFClient] Using 4-bit quantization")
        else:
            model_kwargs["dtype"] = torch_dtype
            model_kwargs["device_map"] = device_map

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

        if device_map is None and not self.quantize:
            model = model.to(device)

        self._model = model
        self._tokenizer = tokenizer

        if device.startswith("cuda"):
            for gpu_id in range(min(num_gpus_to_use, available_gpus)):
                name = torch.cuda.get_device_name(gpu_id)
                gb = torch.cuda.get_device_properties(gpu_id).total_mem / 1024**3 if hasattr(torch.cuda.get_device_properties(gpu_id), 'total_mem') else torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                print(f"[HFClient] GPU {gpu_id}: {name} ({gb:.1f} GB)")

        print("[HFClient] Model loaded successfully.")

    # ---- Message normalization (handles models without system role) ----

    @staticmethod
    def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        system_parts = []
        non_system = []
        for msg in messages:
            if msg.get("role") == "system":
                system_parts.append(str(msg.get("content", "")))
            else:
                non_system.append(dict(msg))

        if not non_system:
            non_system = [{"role": "user", "content": ""}]

        if system_parts:
            prefix = "\n".join(system_parts)
            if non_system[0].get("role") == "user":
                non_system[0]["content"] = f"{prefix}\n\n{non_system[0]['content']}"
            else:
                non_system.insert(0, {"role": "user", "content": prefix})

        merged = []
        for msg in non_system:
            if merged and merged[-1]["role"] == msg["role"]:
                merged[-1]["content"] += "\n\n" + str(msg.get("content", ""))
            else:
                merged.append(msg)

        if merged and merged[0]["role"] != "user":
            merged.insert(0, {"role": "user", "content": ""})

        return merged

    # ---- Prompt building ----

    def _build_prompt(self, prompt) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            messages = list(prompt)
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        if (
            hasattr(self._tokenizer, "chat_template")
            and self._tokenizer.chat_template is not None
        ):
            chat_kwargs = dict(tokenize=False, add_generation_prompt=True)
            if self.enable_thinking is not None:
                chat_kwargs["enable_thinking"] = self.enable_thinking
            try:
                return self._tokenizer.apply_chat_template(
                    messages, **chat_kwargs,
                )
            except TypeError:
                # Tokenizer doesn't support enable_thinking — drop it
                chat_kwargs.pop("enable_thinking", None)
                return self._tokenizer.apply_chat_template(
                    messages, **chat_kwargs,
                )
            except Exception:
                normalized = self._normalize_messages(messages)
                return self._tokenizer.apply_chat_template(
                    normalized, **chat_kwargs,
                )

        user_texts = []
        for msg in messages:
            if msg.get("role") == "user":
                user_texts.append(str(msg.get("content", "")))
        return f"Question: {' '.join(user_texts)}\n\nAnswer:"

    # ---- Generation ----

    def _generate(self, prompt) -> str:
        import torch

        text = self._build_prompt(prompt)

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.truncate_input_tokens,
            padding=False,
        )
        model_device = next(self._model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        temp = float(self.temperature)
        do_sample = temp > 0.0

        gen_kwargs = {
            "max_new_tokens": self.max_tokens,
            "do_sample": do_sample,
            "repetition_penalty": self.repetition_penalty,
            "pad_token_id": self._tokenizer.eos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = self.top_p
            gen_kwargs["top_k"] = self.top_k

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        response_tokens = outputs[0][input_len:]

        # Decode WITH special tokens to detect <think> boundaries
        raw_text = self._tokenizer.decode(response_tokens, skip_special_tokens=False)

        # Strip <think>...</think> blocks (Qwen3.5 thinking mode)
        if '<think>' in raw_text:
            think_end = raw_text.rfind('</think>')
            if think_end != -1:
                # Take only the text AFTER the last </think>
                content = raw_text[think_end + len('</think>'):]
            else:
                # Thinking was cut off by max_tokens — take text before <think>
                content = raw_text.split('<think>')[0]
        else:
            content = raw_text

        # Remove remaining special tokens (e.g., <|im_end|>, <|endoftext|>)
        content = re.sub(r'<\|[^|]*\|>', '', content).strip()

        # Fallback: if stripping left nothing, decode normally
        if not content:
            content = self._tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

        prompt_tokens = int(input_len)
        completion_tokens = int(response_tokens.shape[0])
        self.model_call_counts[self.model_name] += 1
        self.model_input_tokens[self.model_name] += prompt_tokens
        self.model_output_tokens[self.model_name] += completion_tokens
        self.last_prompt_tokens = prompt_tokens
        self.last_completion_tokens = completion_tokens

        return content if content else "[Empty response]"

    def completion(self, prompt) -> str:
        return self._generate(prompt)

    async def acompletion(self, prompt) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate, prompt)

    def print_usage(self):
        print(f"\n[HFClient] Usage for {self.model_name}:")
        calls = self.model_call_counts[self.model_name]
        inp = self.model_input_tokens[self.model_name]
        out = self.model_output_tokens[self.model_name]
        print(f"  Calls: {calls}, Input tokens: {inp}, Output tokens: {out}")
