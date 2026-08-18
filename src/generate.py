"""Lightweight LLM wrapper for Qwen3 on Colab T4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step. "
    "Put the final numeric answer alone on the last line as: #### <answer>"
)


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class LLM:
    def __init__(
        self,
        model_id: str,
        dtype: str = "auto",
        device_map: str = "auto",
        load_in_4bit: bool = False,
        enable_thinking: bool = True,
    ) -> None:
        self.model_id = model_id
        self.enable_thinking = enable_thinking

        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        torch_dtype: Any = "auto"
        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch_dtype,
            quantization_config=quant_config,
        )
        self.model.eval()

    def _build_prompt(self, question: str, prefix: str | None = None) -> str:
        user = question.strip()
        if prefix:
            # Continue from a truncated CoT prefix.
            user = (
                f"{question.strip()}\n\n"
                f"Continue the solution from the incomplete reasoning below. "
                f"Do not restart. Finish with #### <answer>.\n\n"
                f"{prefix.strip()}"
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

        # Qwen3 chat template supports enable_thinking.
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    @torch.inference_mode()
    def generate(
        self,
        question: str,
        *,
        prefix: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> GenerationResult:
        prompt = self._build_prompt(question, prefix=prefix)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p, do_sample=True)
        else:
            gen_kwargs.update(do_sample=False)

        output = self.model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs["input_ids"].shape[-1]
        completion_ids = output[0][prompt_len:]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return GenerationResult(
            text=text.strip(),
            prompt_tokens=int(prompt_len),
            completion_tokens=int(completion_ids.shape[-1]),
        )
