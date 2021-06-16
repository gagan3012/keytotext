#Keytotext Trainer

Keytotext now features a trainer module that can help finetune any model to convert keywords to sentences.

This features many fucntions that are described below:

- ##### Download T5 model from HuggingFace Hub 

```python
def from_pretrained(self, model_name="t5-base"):
    """
    Download Model from HF hub
    :param model_name: T5
    :return: Download the model and tokenizer
    """
```

- ##### Train the Model

