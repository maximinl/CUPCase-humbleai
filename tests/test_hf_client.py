import torch

from hf_client import HFClient


class _FakeTokenizer:
    def __init__(self, raw_text: str, decoded_text: str = ""):
        self.raw_text = raw_text
        self.decoded_text = decoded_text
        self.eos_token_id = 0

    def __call__(self, text, return_tensors=None, truncation=None, max_length=None, padding=None):
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    def decode(self, tokens, skip_special_tokens=False):
        return self.decoded_text if skip_special_tokens else self.raw_text


class _FakeModel:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.last_generate_kwargs = None

    def parameters(self):
        yield self.weight

    def generate(self, **kwargs):
        self.last_generate_kwargs = kwargs
        return torch.tensor([[1, 2, 3, 4]])


def _make_client(raw_text: str, *, thinking_budget=None, decoded_text: str = ""):
    client = HFClient.__new__(HFClient)
    client.model_name = "test-model"
    client.max_tokens = 256
    client.thinking_budget = thinking_budget
    client.temperature = 0.0
    client.top_p = 0.95
    client.top_k = 50
    client.repetition_penalty = 1.2
    client.enable_thinking = True
    client._tokenizer = _FakeTokenizer(raw_text=raw_text, decoded_text=decoded_text)
    client._model = _FakeModel()
    client.truncate_input_tokens = 4096
    client.model_call_counts = {"test-model": 0}
    client.model_input_tokens = {"test-model": 0}
    client.model_output_tokens = {"test-model": 0}
    client.last_prompt_tokens = 0
    client.last_completion_tokens = 0
    client._build_prompt = lambda prompt: "prompt"
    return client


def test_generate_uses_configurable_thinking_budget():
    client = _make_client("<think>reasoning</think>Final answer", thinking_budget=8192)

    result = client._generate("prompt")

    assert result == "Final answer"
    assert client._model.last_generate_kwargs["max_new_tokens"] == 8192


def test_generate_extracts_diagnosis_from_thinking_block_when_answer_missing():
    client = _make_client("<think>The final diagnosis is Keratoacanthoma.</think>")

    result = client._generate("prompt")

    assert result == "Keratoacanthoma"
