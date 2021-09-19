"""SentencePieceUnigramTokenizer and script to train it on a corpus available from Huggingface Datasets."""

import json
import argparse
from typing import Iterator, List, Union

import datasets
