import string
import re
import random
import csv

import torchtext

from utils.bert import BertTokenizer, load_vocab


class DataSetGenerator():
    def __init__(self, vocab_file, max_text_length, mecab_dict=None):
        self.tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=True)
        self.text_field, self.label_field = self._prepare(max_text_length)
        self.vocab, self.ids_to_tokens = self._load_vocab(vocab_file)

    def _prepare(self, max_text_length):
        def _preprocess_IMDb(text):
            '''IMDbの前処理'''
            # 改行コードを消去
            _txt = re.sub('<br />', '', text)

            # カンマ、ピリオド以外の記号をスペースに置換
            for p in string.punctuation:
                if (p == ".") or (p == ","):
                    continue
                else:
                    _txt = text.replace(p, " ")

            # ピリオドなどの前後にはスペースを入れておく
            _txt = text.replace(".", " . ")
            _txt = text.replace(",", " , ")
            return _txt

        def tokenize_IMDb(_txt, tokenizer=self.tokenizer):
            _txt = _preprocess_IMDb(_txt)
            _tokens = tokenizer.tokenize(_txt)
            return _tokens

        _tf = torchtext.data.Field(
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
        _lf = torchtext.data.Field(sequential=False, use_vocab=False)
        return _tf, _lf

    def _load_vocab(self, vocab_file):
        _v, _i2t = load_vocab(vocab_file)
        return _v, _i2t

    def build_vocab(self, train_ds):
        self.text_field.build_vocab(train_ds, min_freq=1)
        self.text_field.vocab.stoi = self.vocab

    def loadTSV(self, tsv_file):
        _ds = torchtext.data.TabularDataset(
            path=tsv_file, format="tsv",
            fields=[('Text', self.text_field), ('Label', self.label_field)])
        return _ds

    def loadTSV_at_index(self, tsv_file, index):
        _fields = [('Text', self.text_field), ('Label', self.label_field)]
        with open(tsv_file, "r") as f:
            _reader = csv.reader(f, delimiter="\t")
            for i, row in enumerate(_reader):
                if i == index:
                    _row = row
                    break
        _data = []
        _data.append(torchtext.data.Example.fromlist(_row, _fields))
        _ds = torchtext.data.Dataset(_data, _fields)
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

    generator = DataSetGenerator(args.vocab_file[0], args.text_length)

    train_validation_ds = generator.loadTSV(args.train_tsv[0])
    train_ds, validation_ds = train_validation_ds.split(split_ratio=0.8, random_state=random.seed(1234))
    generator.build_vocab(train_ds)

    train_dl = get_data_loader(train_ds, args.batch_size, for_train=True)
    validation_dl = get_data_loader(validation_ds, args.batch_size)
    test_dl = get_data_loader(generator.loadTSV(args.test_tsv[0]), args.batch_size)

    batch = next(iter(validation_dl))
    print(batch.Text)
    print(batch.Label)

    # ミニバッチの1文目を確認してみる
    text_minibatch_1 = (batch.Text[0][1]).numpy()
    # IDを単語に戻す
    text = generator.tokenizer.convert_ids_to_tokens(text_minibatch_1)
    print(text)
