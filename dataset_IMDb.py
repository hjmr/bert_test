import string
import re

import torchtext

from utils.bert import BertTokenizer, load_vocab


class FieldSet():
    def __init__(self, vocab_file, max_text_length=256, mecab_dict=None):
        self.tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=True)
        self.text, self.label = self._prepare(max_text_length)
        self.vocab, self.ids_to_tokens = self._load_vocab(vocab_file)

    def _prepare(self, max_text_length):
        def _preprocess_IMDb(text):
            '''IMDbの前処理'''
            # 改行コードを消去
            text = re.sub('<br />', '', text)

            # カンマ、ピリオド以外の記号をスペースに置換
            for p in string.punctuation:
                if (p == ".") or (p == ","):
                    continue
                else:
                    text = text.replace(p, " ")

            # ピリオドなどの前後にはスペースを入れておく
            text = text.replace(".", " . ")
            text = text.replace(",", " , ")
            return text

        def tokenize_IMDb(text, tokenizer=self.tokenizer):
            text = _preprocess_IMDb(text)
            tokens = tokenizer.tokenize(text)
            return tokens

        text_field = torchtext.data.Field(
            sequential=True,
            use_vocab=True,
            fix_length=max_text_length,
            preprocessing=None,
            postprocessing=None,
            lower=True,
            tokenize=tokenize_IMDb,
            include_lengths=True,
            init_token="[CLS]",
            eos_token="[SEP]",
            pad_token="[PAD]",
            unk_token="[UNK]",
            batch_first=True,
            stop_words=None)
        label_field = torchtext.data.Field(sequential=False, use_vocab=False)
        return text_field, label_field

    def _load_vocab(self, vocab_file):
        vocab, ids_to_tokens = load_vocab(vocab_file)
        return vocab, ids_to_tokens

    def build_vocab(self, train_ds):
        self.text.build_vocab(train_ds, min_freq=1)
        self.text.vocab.stoi = self.vocab


def load_data_set(tsv_file, field_set):
    _ds = torchtext.data.TabularDataset(
        path=tsv_file, format="tsv",
        fields=[('Text', field_set.text), ('Label', field_set.label)])
    return _ds


def get_data_loader(data_set, batch_size=16, for_train=False):
    _dl = None
    if for_train:
        _dl = torchtext.data.Iterator(
            data_set, batch_size=batch_size, train=True)
    else:
        _dl = torchtext.data.Iterator(
            data_set, batch_size=batch_size, train=False, sort=False)
    return _dl


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    import argparse

    def parse_arg():
        parser = argparse.ArgumentParser(description="Test DataSet for BERT for Japanese Texts.")
        parser.add_argument("--batch_size", type=int, default=16, help="batch size.")
        parser.add_argument("--text_length", type=int, default=256, help="the length of texts.")
        parser.add_argument("train_tsv", type=str, nargs=1, help="TSV file for train data.")
        parser.add_argument("test_tsv", type=str, nargs=1, help="TSV file for test data.")
        parser.add_argument("vocab_file", type=str, nargs=1, help="a vocabulary file.")
        return parser.parse_args()

    args = parse_arg()

    field_set = FieldSet(args.vocab_file[0], args.text_length)

    train_validation_ds = load_data_set(args.train_tsv[0], field_set)
    train_ds, validation_ds = train_validation_ds.split(split_ratio=0.8)
    field_set.build_vocab(train_ds)

    train_dl = get_data_loader(train_ds, args.batch_size, for_train=True)
    validation_dl = get_data_loader(validation_ds, args.batch_size)
    test_dl = get_data_loader(load_data_set(args.test_tsv[0], field_set), args.batch_size)

    batch = next(iter(validation_dl))
    print(batch.Text)
    print(batch.Label)

    # ミニバッチの1文目を確認してみる
    text_minibatch_1 = (batch.Text[0][1]).numpy()
    # IDを単語に戻す
    text = field_set.tokenizer.convert_ids_to_tokens(text_minibatch_1)
    print(text)
