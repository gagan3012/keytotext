import shutil

import torch
import numpy as np
import pandas as pd
from datasets import load_metric
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
)
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import LightningDataModule
from pytorch_lightning import Trainer
from getpass import getpass
from huggingface_hub import HfApi, Repository
from pathlib import Path

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
            train_df: pd.DataFrame,
            test_df: pd.DataFrame,
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
        self.train_df = train_df
        self.test_df = test_df
        self.split = split
        self.batch_size = batch_size
        self.target_max_token_len = target_max_token_len
        self.source_max_token_len = source_max_token_len
        self.tokenizer = tokenizer

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

    def __init__(self, tokenizer, model, output: str = "outputs"):
        """
        initiates a PyTorch Lightning Model
        Args:
            tokenizer : T5 tokenizer
            model : T5 model
            output (str, optional): output directory to save model checkpoints. Defaults to "outputs".
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.output = output
        # self.val_acc = Accuracy()
        # self.train_acc = Accuracy()

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        """ forward step """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_size):
        """ training step """
        input_ids = batch["keywords_input_ids"]
        attention_mask = batch["keywords_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_size):
        """ validation step """
        input_ids = batch["keywords_input_ids"]
        attention_mask = batch["keywords_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_size):
        """ test step """
        input_ids = batch["keywords_input_ids"]
        attention_mask = batch["keywords_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]

        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels,
        )

        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """ configure optimizers """
        return AdamW(self.parameters(), lr=0.0001)


class trainer:
    """
    Keytotext model trainer
    """

    def __init__(self):
        pass

    def from_pretrained(self, model_name="t5-base"):
        """
        Download Model from HF hub
        :param model_name: T5
        :return: Download the model and tokenizer
        """
        self.tokenizer = T5Tokenizer.from_pretrained(f"{model_name}")
        self.model = T5ForConditionalGeneration.from_pretrained(
            f"{model_name}", return_dict=True
        )

    def train(
            self,
            train_df: pd.DataFrame,
            test_df: pd.DataFrame,
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
        trains T5 model on custom dataset
        Args:
            data_df (pd.DataFrame): training datarame. Dataframe must have 2 column --> "keywords" and "text"
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
            batch_size (int, optional): batch size. Defaults to 8.
            max_epochs (int, optional): max number of epochs. Defaults to 5.
            use_gpu (bool, optional): if True, model uses gpu for training. Defaults to True.
            outputdir (str, optional): output directory to save model checkpoints. Defaults to "outputs".
            early_stopping_patience_epochs (int, optional): monitors val_loss on epoch end and stops training,
             if val_loss does not improve after the specied number of epochs. set 0 to disable early stopping.
             Defaults to 0 (disabled)
            :param test_df:
            :param train_df:
        """
        self.target_max_token_len = target_max_token_len
        self.max_epoch = max_epochs
        self.train_df = train_df
        self.test_df = test_df

        self.data_module = PLDataModule(
            train_df=train_df,
            test_df=test_df,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            source_max_token_len=source_max_token_len,
            target_max_token_len=target_max_token_len,
            split=test_split
        )

        self.T5Model = LightningModel(
            tokenizer=self.tokenizer, model=self.model, output=outputdir
        )

        logger = WandbLogger(project="keytotext")

        early_stop_callback = (
            [
                EarlyStopping(
                    monitor="val_loss",
                    min_delta=0.00,
                    patience=early_stopping_patience_epochs,
                    verbose=True,
                    mode="min",
                )
            ]
            if early_stopping_patience_epochs > 0
            else None
        )

        gpus = -1 if use_gpu else 0

        trainer = Trainer(
            logger=logger,
            callbacks=early_stop_callback,
            max_epochs=max_epochs,
            gpus=gpus,
            progress_bar_refresh_rate=5,
        )

        trainer.fit(self.T5Model, self.data_module)

    def load_model(
            self, model_dir: str = "outputs", use_gpu: bool = False
    ):
        """
        loads a checkpoint for inferencing/prediction
        Args:
            model_dir (str, optional): path to model directory. Defaults to "outputs".
            use_gpu (bool, optional): if True, model uses gpu for inferencing/prediction. Defaults to True.
        """
        self.model = T5ForConditionalGeneration.from_pretrained(f"{model_dir}")
        self.tokenizer = T5Tokenizer.from_pretrained(f"{model_dir}")

        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise Exception("exception ---> no gpu found. set use_gpu=False, to use CPU")
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

    def save_model(
            self,
            model_dir="model"
    ):
        """
        Save model to dir
        :param model_dir:
        :return: model is saved
        """
        path = f"{model_dir}"
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)

    def predict(
            self,
            keywords: list,
            max_length: int = 512,
            num_return_sequences: int = 1,
            num_beams: int = 2,
            top_k: int = 50,
            top_p: float = 0.95,
            do_sample: bool = True,
            repetition_penalty: float = 2.5,
            length_penalty: float = 1.0,
            early_stopping: bool = True,
            skip_special_tokens: bool = True,
            clean_up_tokenization_spaces: bool = True,
            use_gpu: bool = True
    ):
        """
        generates prediction for K2T model
        Args:
            Keywords (list): any keywords for generating predictions
            max_length (int, optional): max token length of prediction. Defaults to 512.
            num_return_sequences (int, optional): number of predictions to be returned. Defaults to 1.
            num_beams (int, optional): number of beams. Defaults to 2.
            top_k (int, optional): Defaults to 50.
            top_p (float, optional): Defaults to 0.95.
            do_sample (bool, optional): Defaults to True.
            repetition_penalty (float, optional): Defaults to 2.5.
            length_penalty (float, optional): Defaults to 1.0.
            early_stopping (bool, optional): Defaults to True.
            skip_special_tokens (bool, optional): Defaults to True.
            clean_up_tokenization_spaces (bool, optional): Defaults to True.
            use_gpu: Defaults to True.
        Returns:
            str: returns predictions
        """
        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise Exception("exception ---> no gpu found. set use_gpu=False, to use CPU")
        else:
            self.device = torch.device("cpu")

        source_text = ' '.join(map(str, keywords))

        input_ids = self.tokenizer.encode(
            source_text, return_tensors="pt", add_special_tokens=True
        )

        input_ids = input_ids.to(self.device)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            num_beams=num_beams,
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
        )
        preds = [
            self.tokenizer.decode(
                g,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
            for g in generated_ids
        ]
        return preds[0]


    def evaluate(
            self,
            test_df: pd.DataFrame,
            metrics: str = "rouge"
    ):
        """

        :param test_df:
        :param metrics:
        :return: Output metrics for keytotext
        """
        metric = load_metric(metrics)
        input_text = test_df['input_text']
        references = test_df['output_text']
        predictions = [self.predict(x) for x in input_text]

        results = metric.compute(predictions=predictions, references=references)

        output = {
            'Rouge 1': {
                'Rouge_1 Low Precision': results["rouge1"].low.precision,
                'Rouge_1 Low recall': results["rouge1"].low.recall,
                'Rouge_1 Low F1': results["rouge1"].low.fmeasure,
                'Rouge_1 Mid Precision': results["rouge1"].mid.precision,
                'Rouge_1 Mid recall': results["rouge1"].mid.recall,
                'Rouge_1 Mid F1': results["rouge1"].mid.fmeasure,
                'Rouge_1 High Precision': results["rouge1"].high.precision,
                'Rouge_1 High recall': results["rouge1"].high.recall,
    def upload(self, hf_username, model_name):
        hf_password = getpass("Enter your HuggingFace password")
        if Path('./model').exists():
            shutil.rmtree('./model')
        token = HfApi().login(username=hf_username, password=hf_password)
        del hf_password
        model_url = HfApi().create_repo(token=token, name=model_name, exist_ok=True)
        model_repo = Repository("./model", clone_from=model_url, use_auth_token=token,
                                git_email=f"{hf_username}@users.noreply.huggingface.co", git_user=hf_username)

        readme_txt = f"""
        ---
language: "en"
thumbnail: "Keywords to Sentences"
tags:
- keytotext
- k2t
- Keywords to Sentences
license: "MIT"
datasets:
- WebNLG
- Dart 
metrics:
- NLG

model-index:
- name: {model_name}
---
        
# keytotext

[![pypi Version](https://img.shields.io/pypi/v/keytotext.svg?logo=pypi&logoColor=white)](https://pypi.org/project/keytotext/)
[![Downloads](https://static.pepy.tech/personalized-badge/keytotext?period=total&units=none&left_color=grey&right_color=orange&left_text=Pip%20Downloads)](https://pepy.tech/project/keytotext)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gagan3012/keytotext/blob/master/notebooks/K2T.ipynb)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/gagan3012/keytotext/UI/app.py)
[![API Call](https://img.shields.io/badge/-FastAPI-red?logo=fastapi&labelColor=white)](https://github.com/gagan3012/keytotext#api)
[![Docker Call](https://img.shields.io/badge/-Docker%20Image-blue?logo=docker&labelColor=white)](https://hub.docker.com/r/gagan30/keytotext)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Models%20on%20Hub-yellow)](https://huggingface.co/models?filter=keytotext)
[![Documentation Status](https://readthedocs.org/projects/keytotext/badge/?version=latest)](https://keytotext.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)



![keytotext](https://socialify.git.ci/gagan3012/keytotext/image?description=1&forks=1&language=1&owner=1&stargazers=1&theme=Light)


Idea is to build a model which will take keywords as inputs and generate sentences as outputs.

Potential use case can include: 
- Marketing 
- Search Engine Optimization
- Topic generation etc.
- Fine tuning of topic modeling models 
        """.strip()

        (Path(model_repo.local_dir) / 'README.md').write_text(readme_txt)
        self.save_model()
        commit_url = model_repo.push_to_hub()

        print("Check out your model at:")
        print(f"https://huggingface.co/{hf_username}/{model_name}")
