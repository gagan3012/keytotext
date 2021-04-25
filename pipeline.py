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
            inputs = _prepare_inputs_for_k2t(keywords)
            if not kwargs:
                kwargs = self.default_generate_kwargs

        def _prepare_inputs_for_k2t(self, keywords):
            text = str(keywords)
            text = text.replace(',', '|')
            text = text.replace("'", "")
            text = text.replace('[', '')
            text = text.replace(']', '')
            texts = text.split(".")

            for txt in texts:
                if self.model_type == "t5":
                    txt = txt + " </s>"
                inputs = self._tokenize(txt, padding=False)
            return inputs

        def _tokenize(
                self,
                inputs,
                padding=True,
                truncation=True,
                add_special_tokens=True,
                max_length=1024
        ):
            inputs = self.tokenizer.batch_encode_plus(
                inputs,
                max_length=max_length,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                padding="max_length" if padding else False,
                pad_to_max_length=padding,
                return_tensors="pt"
            )
            return inputs
