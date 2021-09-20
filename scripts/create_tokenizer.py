# coding: utf-8

"""SentencePieceUnigramTokenizer and script to train it on a corpus available from Huggingface Datasets."""

import json
import argparse
from typing import Iterator, List, Union

import datasets
from datasets.utils import DownloadConfig
from transformers import T5Config

from tokenizers import (
    AddedToken,
    Regex,
from tokenizers.implementations.base_tokenizer import BaseTokenizer
from tokenizers.models import Unigram
from tokenizers.processors import TemplateProcessing


class SentencePieceUnigramTokenizer(BaseTokenizer):
    """
    This class is a copy of `DeDLOC's tokenizer implementation <https://github.com/yandex-research/DeDLOC/blob/main/sahajbert/tokenizer/tokenizer_model.py>`
    Custom SentencePiece Unigram Tokenizer with NMT, NKFC, spaces and lower-casing characters normalization
    Represents the Unigram algorithm, with the pretokenization used by SentencePiece
    """
    def __init__(
        self,
        replacement: str = "▁",
        add_prefix_space: bool = True,
        unk_token: Union[str, AddedToken] = "<unk>",
        eos_token: Union[str, AddedToken] = "</s>",
        pad_token: Union[str, AddedToken] = "<pad>",
    ):
        self.special_tokens = {
            "pad": {"id": 0, "token": pad_token},
            "eos": {"id": 1, "token": eos_token},
            "unk": {"id": 2, "token": unk_token},
        }
        self.special_tokens_list = [None] * len(self.special_tokens)
        for token_dict in self.special_tokens.values():
            self.special_tokens_list[token_dict["id"]] = token_dict["token"]
        tokenizer = Tokenizer(Unigram())
        tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.Nmt(),
                normalizers.NFKC(),
                normalizers.Replace(Regex(" {2,}"), " "),
                normalizers.Lowercase(),
            ]
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space),
                pre_tokenizers.Digits(individual_digits=True),
                pre_tokenizers.Punctuation(),
            ]
        )
        tokenizer.decoder = decoders.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space)
        tokenizer.post_processor = TemplateProcessing(
            single=f"$A {self.special_tokens['eos']['token']}",
            special_tokens=[(self.special_tokens["eos"]["token"], self.special_tokens["eos"]["id"])],
        )
        parameters = {
            "model": "SentencePieceUnigram",
            "replacement": replacement,
            "add_prefix_space": add_prefix_space,
        }
        super().__init__(tokenizer, parameters)

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 8000,
        show_progress: bool = True,
    ):
        """Train the model using the given files"""
        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=self.special_tokens_list,
            show_progress=show_progress,
        )
        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(files, trainer=trainer)
        self.add_unk_id()

    def train_from_iterator(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        vocab_size: int = 8000,
        show_progress: bool = True,
    ):
        """Train the model using the given iterator"""
        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=self.special_tokens_list,
            show_progress=show_progress,
        )
        self._tokenizer.train_from_iterator(iterator, trainer=trainer)
        self.add_unk_id()

    def add_unk_id(self):
        tokenizer_json = json.loads(self._tokenizer.to_str())
        tokenizer_json["model"]["unk_id"] = self.special_tokens["unk"]["id"]
        self._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))


def main(args):
    # Initialize a dataset
    cfg = DownloadConfig(num_proc=args.num_proc)
    dataset = datasets.load_dataset(
        args.dataset,
        name=args.dataset_config,
        split=args.dataset_split,
        download_config=cfg
    )
    # Initialize a tokenizer
    tokenizer = SentencePieceUnigramTokenizer(
        unk_token="<unk>", eos_token="</s>", pad_token="<pad>"
    )
    # Build an iterator over this dataset
    def batch_iterator(input_sentence_size=None, batch_size=args.batch_size):
        if input_sentence_size is None:
            input_sentence_size = len(dataset)
        for i in range(0, input_sentence_size, batch_size):
            yield dataset[i: i + batch_size][args.text_field]
    # Train tokenizer
    tokenizer.train_from_iterator(
        iterator=batch_iterator(input_sentence_size=args.input_sentence_size),
        vocab_size=args.vocab_size,
        show_progress=True,
    )
    # Save files to disk
    tokenizer.save(f"{args.model_dir}/tokenizer.json")
    config = T5Config.from_pretrained(
        args.config_type, vocab_size=tokenizer.get_vocab_size()
    )
    config.save_pretrained(args.model_dir)
    print(f"Tokenizer and config saved in {args.model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--vocab_size", type=int, default=32_000)
    parser.add_argument("--input_sentence_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--num_proc", type=int, default=96)
    parser.add_argument("--config_type", type=str, default="google/t5-v1_1-base")
    args = parser.parse_args()
    main(args)