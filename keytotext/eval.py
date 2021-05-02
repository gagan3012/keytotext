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
    model_type: str = field(
        metadata={"help": "T5"}
    )
    output_path: Optional[str] = field(
        default="hypothesis.txt",
        metadata={"help": "path to save the generated text from keywords."}
    )




