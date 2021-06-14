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
        :param split:
        """
        super().__init__()
        self.data_df = data_df
        self.split = split
        self.batch_size = batch_size
        self.target_max_token_len = target_max_token_len
        self.source_max_token_len = source_max_token_len
        self.tokenizer = tokenizer

    def prepare_data(self):
        self.train_df, self.test_df = train_test_split(self.data_df, test_size=self.split, shuffle=True)

    def setup(self, stage=None):
        self.train_dataset = DataModule(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )
        self.test_dataset = DataModule(
            self.test_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len,
        )

    def train_dataloader(self):
        """ training dataloader """
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

    def test_dataloader(self):
        """ test dataloader """
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

    def val_dataloader(self):
        """ validation dataloader """
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

class LightningModel(pl.LightningModule):
    """ PyTorch Lightning Model class"""

    def __init__(self, tokenizer, model, outputdir: str = "outputs"):
        """
        initiates a PyTorch Lightning Model
        Args:
            tokenizer : T5/MT5 tokenizer
            model : T5/MT5 model
            outputdir (str, optional): output directory to save model checkpoints. Defaults to "outputs".
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.outputdir = outputdir

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        """ forward step """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        return output.loss, output.logits

