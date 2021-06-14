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
from pytorch_lightning.loggers import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

torch.cuda.empty_cache()
pl.seed_everything(42)

class DataModule(Dataset):

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: T5Tokenizer,
        source_max_token_len: int = 512,
        target_max_token_len: int = 512,
    ):
        self.data = data
        self.target_max_token_len = target_max_token_len
        self.source_max_token_len = source_max_token_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index: int):
        
