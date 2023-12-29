import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from llama_cpp import Llama as LlamaCpp

from opencompass.models.base import BaseModel
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

@MODELS.register_module()
class LlamaCppModel(BaseModel):

    def __init__(self,
                 model_path: str,
                 max_seq_len: int = 2048,
                 meta_template: Optional[Dict] = None,
                 **kwargs):
        super().__init__(
            path=model_path,
            max_seq_len=max_seq_len,
            meta_template=meta_template,
        )
        self.llama_cpp = LlamaCpp(
            model_path,
            n_ctx=max_seq_len,
            n_gpu_layers=1024,
            offload_kqv=True,
        )

    # def parser

    def get_token_len(self, prompt: str) -> int:
        tokens = self.llama_cpp.tokenize(prompt.encode('utf-8'))
        print(f"prompt [{prompt}] encoded to [{tokens}]")
        return len(tokens)

    def generate(self, inputs: List[str], max_out_len: int, temperature: float=0.0) -> List[str]:
        assert (len(inputs) == 1)
        output = self.llama_cpp.create_completion(
            prompt=inputs[0],
            max_tokens=max_out_len,
            temperature=temperature,
            stop=None,
        )
        return [output['choices'][0]['text']]

    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        raise NotImplementedError

if __name__ == '__main__':
    model_path = "/home/wangyin.yx/models/gguf/qwen-7b-chat-q8.gguf"
    model = LlamaCppModel(model_path)
    query = "Hi, "
    print(model.generate([query], 1024))
