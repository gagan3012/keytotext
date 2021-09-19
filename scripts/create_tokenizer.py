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
from tokenizers.processors import TemplateProcessing


class SentencePieceUnigramTokenizer(BaseTokenizer):
    """
    This class is a copy of `DeDLOC's tokenizer implementation <https://github.com/yandex-research/DeDLOC/blob/main/sahajbert/tokenizer/tokenizer_model.py>`__ .
    Custom SentencePiece Unigram Tokenizer with NMT, NKFC, spaces and lower-casing characters normalization
    Represents the Unigram algorithm, with the pretokenization used by SentencePiece
    """
    def __init__(
        self,
