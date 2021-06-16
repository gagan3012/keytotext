#Keytotext Trainer

Keytotext now features a trainer module that can help finetune any model to convert keywords to sentences.

This features many fucntions that are described below:

- ### Download T5 model from HuggingFace Hub 

```python
def from_pretrained(self, model_name="t5-base"):
