import re
import argparse

from normalize_text import normalize_text


def parse_arg():
    parser = argparse.ArgumentParser(description="Remove URL and emtpy lines.")
    parser.add_argument("--min_length", type=int, default=10, help="the minimum length of line.")
    parser.add_argument("tsv_file", type=str, help="TSV file.")
    return parser.parse_args()


def run_main():
    args = parse_arg()
    with open(args.tsv_file, "r") as f:
        for line in f:
            tokens = line.strip().split("\t")
            text = " ".join(tokens[:-1])
            text = normalize_text(text)
            label = tokens[-1]
            if args.min_length <= len(text):
                print("{}\t{}".format(text, label))


if __name__ == "__main__":
    run_main()
