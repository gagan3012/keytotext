from typing import Optional, Union
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from .models import NMPipeline, K2TPipeline


SUPPORTED_TASKS = {
    "k2t": {
        "impl": K2TPipeline,
        "default": {
            "model": "gagan3012/k2t",
        },
    },
    "k2t-base": {
        "impl": K2TPipeline,
        "default": {
            "model": "gagan3012/k2t-base",
        },
    },
    "mrm8488/t5-base-finetuned-common_gen": {
        "impl": NMPipeline,
        "default": {
            "model": "mrm8488/t5-base-finetuned-common_gen",
        },
    },
    "k2t-new": {
        "impl": NMPipeline,
        "default": {
            "model": "gagan3012/k2t-new",
        },
    }
}


def pipeline(
        task: str,
        model: Optional = None,
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        use_cuda: Optional[bool] = True,
):
    """

    :param task:
    (:obj:`str`):
            The task defining which pipeline will be returned. Currently accepted tasks are:
            - :obj:`"k2t"`: will return a :class:`K2TPipeline` which is based on the k2t model based on t5-small
            - :obj:`"k2t-tiny"`: will return a :class:`K2TPipeline` which is based on the k2t model based on t5-tiny
            - :obj:`"k2t-base"`: will return a :class:`K2TPipeline` which is based on the k2t model based on t5-base
    :param model:
    (:obj:`str` or `optional`):
            The model that will be used by the pipeline to make predictions.

            If not provided, the default for the :obj:`task` will be loaded.
    :param tokenizer:
    (:obj:`str` or `optional`):
            The tokenizer that will be used by the pipeline to encode data for the model. This can be a model
            identifier or an actual pretrained tokenizer inheriting from :class:`~transformers.PreTrainedTokenizer`.

            If not provided, the default tokenizer for the given :obj:`model` will be loaded (if it is a string).
    :param use_cuda:
    (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to use a GPU or not Default: True
    :return:
    (:class:):
            `K2TPipeline`: A Keytotext pipeline for the task.

    """

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

    return task_class(model=model, tokenizer=tokenizer, use_cuda=use_cuda)


