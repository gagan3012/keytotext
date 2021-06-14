import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
    AutoModelWithLMHead
