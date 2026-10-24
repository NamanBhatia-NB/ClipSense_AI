from typing import Optional
from argparse import Namespace
from datasets import Dataset
from methods.base import BaseMethod
from methods.heuristics import RandomMethod, FrequencyMethod
from methods.bm25 import BM25Method
from methods.zeroshot import ZeroShotMethod
from methods.ft_llm import FtLlmMethod

METHOD_REGISTRY = {
    "random": RandomMethod,
    "frequency": FrequencyMethod,
    "bm25": BM25Method,
    "zeroshot": ZeroShotMethod,
    "ft_llm": FtLlmMethod,
}

def list_methods() -> list[str]:
    return list[str](METHOD_REGISTRY.keys())

def get_method(
    method_name: str,
    args: Optional[Namespace] = None,
    train_dataset: Optional[Dataset] = None,
    return_cls: bool = False,
) -> BaseMethod:
    if return_cls:
        return METHOD_REGISTRY[method_name]
    return METHOD_REGISTRY[method_name](args, train_dataset)
