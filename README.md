# BERTによるテキスト分類

BERTを用いてテキストの分類をするための一連のプログラム。

## 1. 準備

### 1-1. MeCabのインストール

- 分かち書きをするためにMeCabを利用しているので，MeCabをインストールしておく。
- 必要であればMeCabの辞書（neologdなど）をインストールしておく。

### 1-2. モデルの準備

以下のファイルを備えたモデルを用意
- 設定ファイル：ネットワークの構造やパラメータが記述されたJSON形式のファイル
- ボキャブラリファイル：利用する単語が並べられたテキストファイル
- 事前学習済みのBERTモデル：PyTorch形式で保存されたモデルファイル（torch.loadで読めるもの）

手元では以下のモデルで動作を確認している。
- [京大・黒橋研のモデル](http://nlp.ist.i.kyoto-u.ac.jp/index.php?BERT日本語Pretrainedモデル)（日本語）
- [Googleの多言語モデル](https://github.com/google-research/bert/blob/master/multilingual.md)（多言語）
- [Hugging Faceモデル](https://huggingface.co/transformers/pretrained_models.html)
    - いろいろあるけど試したのは ``bert-base-uncased``（英語）

以下のモデルは使えそうだけど未チェック
- [Laboro.AI BERTモデル](https://laboro.ai/column/laboro-bert/)（多分日本語）

### 1-3. 学習用・テスト用データの準備

データはTSV（タブ区切りのテキストファイル）で用意して，1行に「テキストデータ」と「ラベル」をタブ区切りで並べる。
なお，日本語であっても分かち書きする必要はない。
以下は例（数値の前の空白はタブ）

    For a movie that gets no respect         0
    Bizarre horror movie filled with famous faces     0
    日本ホラー映画の先駆けともいえる映画。   1

## 2. 訓練

### train.py

#### 利用法

以下のように実行する。

``` shell
python3 train.py [<options>] <model_config.json> <model.bin> <train_data.tsv> <model_vocab.txt>
```

最低限必要な引数は以下の通りである。
- **model_config.json** : 設定ファイル（モデル）
- **model.bin** : 事前学習済みBERTモデル（モデル）
- **train_data.tsv** : 学習用TSVデータ
- **model_vocab.txt** : ボキャブラリファイル（モデル）

実際の実行は以下のように行う。

``` shell
python3 train.py --batch_size 16 --text_length 256 --epoch 100 --save_path ./results/ ./model/bert_config.json ./model/pytorth_model.bin ./data/train.tsv ./model/vocab.txt
```

Poetryから実行する場合は

``` shell
poetry run python  train.py --batch_size 16 --text_length 256 --epoch 100 --save_path ./results/ ./model/bert_config.json ./model/pytorth_model.bin ./data/train.tsv ./model/vocab.txt
```

#### オプション

- **--mecab_dict** : MeCabで標準以外の辞書を使う場合に辞書の場所を指定する。
- **--batch_size** : バッチサイズを指定（デフォルト：16）
- **--text_length** : BERTに一度に入力するテキストの単語数（デフォルト：256）
- **--random_seed** : 乱数の種を指定
- **--epoch** : 学習のループ回数を指定
- **--save_path** : 学習後のデータ等を保存するディレクトリを指定
- **--IMDb** : IMDbのデータを利用する場合に指定

## 3. テスト

