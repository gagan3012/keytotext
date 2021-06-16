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

```python
    def train(
            self,
            data_df: pd.DataFrame,
            source_max_token_len: int = 512,
            target_max_token_len: int = 512,
            batch_size: int = 8,
            max_epochs: int = 5,
            use_gpu: bool = True,
            outputdir: str = "outputs",
            early_stopping_patience_epochs: int = 0,  # 0 to disable early stopping feature
            test_split=0.1,
    ):
        """
