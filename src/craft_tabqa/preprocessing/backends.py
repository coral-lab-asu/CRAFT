"""Text-generation backends for the LLM enrichment step.

All backends expose the same tiny interface::

    backend.generate(prompts: list[str]) -> list[str]

so the enrichment code never cares whether generation runs through vLLM, plain
transformers, or a hosted API. Pick one with :func:`make_backend`.

vLLM is the default and by far the fastest for corpus-scale generation; the
transformers backend is a dependency-light fallback, and the OpenAI backend is
there when you would rather call a hosted model.
"""

from typing import List


class GenerationBackend:
    """Interface every backend implements."""

    def generate(self, prompts: List[str]) -> List[str]:
        raise NotImplementedError


class VLLMBackend(GenerationBackend):
    """Batched local generation with vLLM (recommended for large corpora)."""

    def __init__(self, model: str, max_new_tokens: int, temperature: float, hf_cache: str = ""):
        from vllm import LLM, SamplingParams

        download_dir = hf_cache or None
        self._llm = LLM(model=model, download_dir=download_dir)
        self._params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature)

    def generate(self, prompts: List[str]) -> List[str]:
        outputs = self._llm.generate(prompts, self._params)
        # vLLM may reorder internally; each output carries its prompt, but the
        # returned list is already aligned to the input order.
        return [out.outputs[0].text.strip() for out in outputs]


class TransformersBackend(GenerationBackend):
    """Plain HuggingFace ``transformers`` generation (dependency-light fallback)."""

    def __init__(self, model: str, max_new_tokens: int, temperature: float, hf_cache: str = ""):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = {"cache_dir": hf_cache} if hf_cache else {}
        self._tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        self._model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype="auto", device_map="auto", **kwargs
        )
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._torch = torch

    def generate(self, prompts: List[str]) -> List[str]:
        outputs: List[str] = []
        for prompt in prompts:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            with self._torch.no_grad():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=self._max_new_tokens,
                    temperature=self._temperature,
                    do_sample=self._temperature > 0,
                )
            new_tokens = generated[0][inputs["input_ids"].shape[1]:]
            outputs.append(self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        return outputs


class OpenAIBackend(GenerationBackend):
    """Hosted generation through the OpenAI chat completions API."""

    def __init__(self, model: str, max_new_tokens: int, temperature: float):
        import os

        from openai import OpenAI

        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._model = model
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

    def generate(self, prompts: List[str]) -> List[str]:
        outputs: List[str] = []
        for prompt in prompts:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self._max_new_tokens,
                temperature=self._temperature,
            )
            outputs.append(resp.choices[0].message.content.strip())
        return outputs


def make_backend(
    backend: str,
    model: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    hf_cache: str = "",
) -> GenerationBackend:
    """Construct the requested backend (``"vllm"``, ``"transformers"``, ``"openai"``)."""
    if backend == "vllm":
        return VLLMBackend(model, max_new_tokens, temperature, hf_cache)
    if backend == "transformers":
        return TransformersBackend(model, max_new_tokens, temperature, hf_cache)
    if backend == "openai":
        return OpenAIBackend(model, max_new_tokens, temperature)
    raise ValueError(f"Unknown backend {backend!r} (use 'vllm', 'transformers', or 'openai')")
