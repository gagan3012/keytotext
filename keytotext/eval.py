import re
from typing import Optional, Union
import torch
from dataclasses import dataclass, field
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser
)


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


def eval(model, tokenizer, keywords, use_cuda):
    result = ""
    predictions = []
    device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"

    for txt in keywords:
        input_ids = _tokenize(tokenizer=tokenizer, inputs="{} </s>".format(txt), padding=False)
        outputs = model.generate(input_ids.to(device))
        result += tokenizer.decode(outputs[0])

    result = re.sub("<pad>|</s>", "", result)

    predictions.extend(result.strip())
    return predictions


def _tokenize(
        tokenizer,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=1024,
):
    inputs = tokenizer.encode(
        inputs,
        max_length=max_length,
        add_special_tokens=add_special_tokens,
        truncation=truncation,
        padding="max_length" if padding else False,
        pad_to_max_length=padding,
        return_tensors="pt",
    )
    return inputs
