import argparse
from transformers import T5ForConditionalGeneration, TFT5ForConditionalGeneration


def main(args):
    pt_model = T5ForConditionalGeneration.from_pretrained(
        args.model_dir, from_flax=True
    )
    pt_model.save_pretrained(args.model_dir)
    tf_model = TFT5ForConditionalGeneration.from_pretrained(args.model_dir, from_pt=True)
    tf_model.save_pretrained(args.model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='.')