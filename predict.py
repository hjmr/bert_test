import argparse

import torch

from utils.bert import BertModel, get_config
# from dataset_jp_text import FieldSet, load_data_set, get_data_loader
from dataset_IMDb import FieldSet, load_data_set, get_data_loader
from bert_cls import BertClassifier


def parse_arg():
    parser = argparse.ArgumentParser(description="Predict using BERT model.")
    parser.add_argument("--mecab_dict", type=str, help="MeCab dictionary.")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size.")
    parser.add_argument("--text_length", type=int, default=256, help="the length of texts.")
    #
    parser.add_argument("--index", type=int, default=0, help="index of the text to be predicted.")
    parser.add_argument("--save_html", type=str, help="output HTML file.")
    #
    parser.add_argument("conf", type=str, nargs=1, help="a BERT configuration file.")
    parser.add_argument("load_path", type=str, nargs=1, help="a path to trained net.")
    #
    parser.add_argument("tsv_file", type=str, nargs=1, help="TSV file for test.")
    parser.add_argument("vocab_file", type=str, nargs=1, help="a vocabulary file.")
    return parser.parse_args()


def highlight(word, attn):
    "Attentionの値が大きいと文字の背景が濃い赤になるhtmlを出力させる関数"
    html_color = '#%02X%02X%02X' % (
        255, int(255*(1 - attn)), int(255*(1 - attn)))
    return '<span style="background-color: {}"> {}</span>'.format(html_color, word)


def mk_html(index, batch, preds, normlized_weights, tokenizer):
    "HTMLデータを作成する"

    # indexの結果を抽出
    sentence = batch.Text[0][index]  # 文章
    label = batch.Label[index]  # ラベル
    pred = preds[index]  # 予測

    # ラベルと予測結果を文字に置き換え
    if label == 0:
        label_str = "Negative"
    else:
        label_str = "Positive"

    if pred == 0:
        pred_str = "Negative"
    else:
        pred_str = "Positive"

    # 表示用のHTMLを作成する
    html = '正解ラベル：{}<br>推論ラベル：{}<br><br>'.format(label_str, pred_str)

    # Self-Attentionの重みを可視化。Multi-Headが12個なので、12種類のアテンションが存在
    for i in range(12):

        # indexのAttentionを抽出と規格化
        # 0単語目[CLS]の、i番目のMulti-Head Attentionを取り出す
        # indexはミニバッチの何個目のデータかをしめす
        attens = normlized_weights[index, i, 0, :]
        attens /= attens.max()

        html += '[BERTのAttentionを可視化_' + str(i+1) + ']<br>'
        for word, attn in zip(sentence, attens):

            # 単語が[SEP]の場合は文章が終わりなのでbreak
            if tokenizer.convert_ids_to_tokens([word.numpy().tolist()])[0] == "[SEP]":
                break

            # 関数highlightで色をつける、関数tokenizer_bert.convert_ids_to_tokensでIDを単語に戻す
            html += highlight(tokenizer.convert_ids_to_tokens(
                [word.numpy().tolist()])[0], attn)
        html += "<br><br>"

    # 12種類のAttentionの平均を求める。最大値で規格化
    # all_attens = attens*0  # all_attensという変数を作成する
    for i in range(12):
        attens += normlized_weights[index, i, 0, :]
    attens /= attens.max()

    html += '[BERTのAttentionを可視化_ALL]<br>'
    for word, attn in zip(sentence, attens):

        # 単語が[SEP]の場合は文章が終わりなのでbreak
        if tokenizer.convert_ids_to_tokens([word.numpy().tolist()])[0] == "[SEP]":
            break

        # 関数highlightで色をつける、関数tokenizer_bert.convert_ids_to_tokensでIDを単語に戻す
        html += highlight(tokenizer.convert_ids_to_tokens(
            [word.numpy().tolist()])[0], attn)
    html += "<br><br>"

    return html


def predict(net, inputs):
    outputs, attention_probs = net(
        inputs,
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=False,
        attention_show_flg=True)

    _, preds = torch.max(outputs, 1)
    return preds, attention_probs


def run_main():
    args = parse_arg()

    print("1. preparing datasets ... ", end="", flush=True)
    field_set = FieldSet(args.vocab_file[0], args.text_length, args.mecab_dict)
    test_ds = load_data_set(args.tsv_file[0], field_set)
    test_dl = get_data_loader(test_ds, args.batch_size, for_train=False)
    field_set.build_vocab(test_ds)
    print("done.", flush=True)

    print("2. loading network ... ", end="", flush=True)
    conf = get_config(file_path=args.conf[0])
    bert_base = BertModel(conf)
    net = BertClassifier(bert_base, out_features=2)
    net.load_state_dict(torch.load(args.load_path[0]))
    net.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print("done.", flush=True)

    print("3. predicting.", flush=True)
    batch = next(iter(test_dl))
    inputs = batch.Text[0].to(device)  # 文章
    preds, attention_probs = predict(net, inputs)

    html = mk_html(args.index, batch, preds, attention_probs, field_set.tokenizer)
    if args.save_html is not None:
        with open(args.save_html, "w") as f:
            f.write(html)
    else:
        print(html)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    run_main()
