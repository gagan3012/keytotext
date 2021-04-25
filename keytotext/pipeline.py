import re
from typing import Optional, Union
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class K2TPipeline:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,

        use_cuda: bool
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)

        assert self.model.__class__.__name__ in ["T5ForConditionalGeneration"]

        if "T5ForConditionalGeneration" in self.model.__class__.__name__:
            self.model_type = "t5"

        self.default_generate_kwargs = {
            "max_length": 1024,
            "num_beams": 4,
            "length_penalty": 1.5,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }

    def __call__(self, keywords, **kwargs):
        inputs = self._prepare_inputs_for_k2t(keywords)
        result = ""
        if not kwargs:
            kwargs = self.default_generate_kwargs

        for txt in inputs:
            input_ids = self._tokenize("{} </s>".format(txt), padding=False)
            outputs = self.model.generate(input_ids.to(self.device), **kwargs)
            result += self.tokenizer.decode(outputs[0])

        result = re.sub("<pad>|</s>", "", result)
        return result

    def _prepare_inputs_for_k2t(self, keywords):
        text = str(keywords)
        text = text.replace(",", "|")
        text = text.replace("'", "")
        text = text.replace("[", "")
        text = text.replace("]", "")
        texts = text.split(".")
        return texts

    def _tokenize(
        self,
        inputs,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        max_length=1024,
    ):
        inputs = self.tokenizer.encode(
            inputs,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt",
        )
        return inputs


SUPPORTED_TASKS = {
    "k2t": {
        "impl": K2TPipeline,
        "default": {
            "model": "gagan3012/k2t",
        },
    }
}


def pipeline(
    task: str,
    model: Optional = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    use_cuda: Optional[bool] = True,
):
    if task not in SUPPORTED_TASKS:
        raise KeyError(
            "Unknown task {}, available tasks are {}".format(
                task, list(SUPPORTED_TASKS.keys())
            )
        )

    targeted_task = SUPPORTED_TASKS[task]
    task_class = targeted_task["impl"]

    if model is None:
        model = targeted_task["default"]["model"]

    if tokenizer is None:
        if isinstance(model, str):
            tokenizer = model
        else:
            # Impossible to guest what is the right tokenizer here
            raise Exception(
                "Please provided a PretrainedTokenizer "
                "class or a path/identifier to a pretrained tokenizer."
            )
    if isinstance(tokenizer, (str, tuple)):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    # Instantiate model if needed
    if isinstance(model, str):
        model = AutoModelForSeq2SeqLM.from_pretrained(model)

    if task == "k2t":
        return task_class(model=model, tokenizer=tokenizer, use_cuda=use_cuda)
