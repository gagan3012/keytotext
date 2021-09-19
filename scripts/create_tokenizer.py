"""SentencePieceUnigramTokenizer and script to train it on a corpus available from Huggingface Datasets."""

import json
import argparse
from typing import Iterator, List, Union

import datasets
from datasets.utils import DownloadConfig
from transformers import T5Config

from tokenizers import AddedToken, Regex, Tokenizer, decoders, normalizers, pre_tokenizers, trainers
from tokenizers.implementations.base_tokenizer import BaseTokenizer
from tokenizers.models import Unigram
