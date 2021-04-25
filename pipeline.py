from transformers import(
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

class K2TPipeline:
    def __init__(
        self,
        model : PreTrainedModel,
        tokenizer : PreTrainedTokenizer,
        use_cuda: bool
    ):

