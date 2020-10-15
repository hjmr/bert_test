import csv

import MeCab
import torchtext

from utils.bert import BertTokenizer, load_vocab


class DataSetGenerator():
    def __init__(self, vocab_file, max_text_length=256, use_basic_form=False, mecab_dict=None):
        self.tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=False, do_basic_tokenize=False)
        if mecab_dict is not None:
            self.tagger = MeCab.Tagger("-d {}".format(mecab_dict))
        else:
            self.tagger = MeCab.Tagger("")
        self.text_field, self.label_field = self._prepare(max_text_length, use_basic_form)
        self.vocab, self.ids_to_tokens = self._load_vocab(vocab_file)

    def _prepare(self, max_text_length, use_basic_form):
        def _han2zen(_text):
            _zenkaku_list = "".join(chr(0xff01 + i) for i in range(94))
            _hankaku_list = "".join(chr(0x21 + i) for i in range(94))
            _han2zen_table = str.maketrans(_hankaku_list, _zenkaku_list)
            _text = _text.translate(_han2zen_table)
            return _text

        def _wakati_jp_text(_text):
            self.tagger.parse("")  # to avoid bug

            _word_list = []
            _token = self.tagger.parseToNode(_text.strip())
            while _token:
                _features = _token.feature.split(",")
                if _features[0] == "記号" and _features[1] == "句点":
                    _word_list.append(".")
                elif _features[0] == "記号" and _features[1] == "読点":
                    _word_list.append(",")
                else:
                    if use_basic_form:
                        _word_list.append(_features[6] if 0 < len(_features[6]) else _token.surface)
                    else:
                        _word_list.append(_token.surface)
                _token = _token.next
            _wakati = " ".join(_word_list)
            return _wakati

        def tokenize_jp(_text, tokenizer=self.tokenizer):
            _zenkaku_text = _han2zen(_text)
            _wakati = _wakati_jp_text(_zenkaku_text)
            _tokens = tokenizer.tokenize(_wakati)
            return _tokens

        text_field = torchtext.data.Field(
            sequential=True,
            use_vocab=True,
            fix_length=max_text_length,
            preprocessing=None,
            postprocessing=None,
            lower=False,
            tokenize=tokenize_jp,
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
        parser.add_argument("--mecab_dict", type=str, help="MeCab dictionary.")
        parser.add_argument("--batch_size", type=int, default=16, help="batch size.")
        parser.add_argument("--text_length", type=int, default=256, help="the length of texts.")
        parser.add_argument("--index", type=int, help="index of text to be shown.")
        parser.add_argument("tsv_file", type=str, nargs=1, help="a TSV file.")
        parser.add_argument("vocab_file", type=str, nargs=1, help="a vocabulary file.")
        return parser.parse_args()

    args = parse_arg()

    generator = DataSetGenerator(args.vocab_file[0], args.text_length, mecab_dict=args.mecab_dict)

    if args.index is not None:
        data_set = generator.loadTSV_at_index(args.tsv_file[0], args.index)
        generator.build_vocab(data_set)
        data_loader = get_data_loader(data_set)
        batch = next(iter(data_loader))
        print(batch)
        print(batch.Text, batch.Label)
        text_minibatch = (batch.Text[0][0]).numpy()
        text = generator.tokenizer.convert_ids_to_tokens(text_minibatch)
        print(text)
    else:
        data_set = generator.loadTSV(args.tsv_file[0])
        generator.build_vocab(data_set)
        data_loader = get_data_loader(data_set, args.batch_size)

        batch = next(iter(data_loader))
        print(batch.Text[0])
        print(batch.Label)

        # ミニバッチの1文目を確認してみる
        text_minibatch = (batch.Text[0][0]).numpy()
        # IDを単語に戻す
        text = generator.tokenizer.convert_ids_to_tokens(text_minibatch)
        print(text)
