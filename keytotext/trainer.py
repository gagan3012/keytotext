import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
    AutoTokenizer,
    AutoModelWithLMHead
)
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import LightningDataModule

torch.cuda.empty_cache()
pl.seed_everything(42)


class DataModule(Dataset):
    """
    Data Module for pytorch
    """

    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: T5Tokenizer,
            source_max_token_len: int = 512,
            target_max_token_len: int = 512,
    ):
        """

        :param data:
        :param tokenizer:
        :param source_max_token_len:
        :param target_max_token_len:
        """
        self.data = data
        self.target_max_token_len = target_max_token_len
        self.source_max_token_len = source_max_token_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        keywords_encoding = self.tokenizer(
            data_row["keywords"],
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        text_encoding = self.tokenizer(
            data_row["text"],
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        labels = text_encoding["input_ids"]
        labels[
            labels == 0
            ] = -100

        return dict(
            keywords=data_row["keywords"],
            text=data_row["text"],
            keywords_input_ids=keywords_encoding["input_ids"].flatten(),
            keywords_attention_mask=keywords_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=text_encoding["attention_mask"].flatten(),
        )


class PLDataModule(LightningDataModule):
    def __init__(
            self,
            data_df: pd.DataFrame,
            tokenizer: T5Tokenizer,
            source_max_token_len: int = 512,
            target_max_token_len: int = 512,
            batch_size: int = 4,
            split: float = 0.1
    ):
        """

        :param data_df:
        :param tokenizer:
        :param source_max_token_len:
        :param target_max_token_len:
        :param batch_size:
        super().__init__()
        self.test_df = test_df
        self.train_df = train_df
        self.split = split
        self.batch_size = batch_size
        self.target_max_token_len = target_max_token_len
        self.source_max_token_len = source_max_token_len
        self.tokenizer = tokenizer

    def setup(self, ):


