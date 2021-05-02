import re
from typing import Optional, Union
import torch
from dataclasses import dataclass, field
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser
)

device = 'cuda' if torch.cuda.is_available else 'cpu'

@dataclass
class EvalArgs:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model"}
    )
    model_type: str = field(metadata={"help": "One of 't5', 'bart'"})




