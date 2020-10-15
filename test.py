import argparse
from tqdm import tqdm

import torch

from utils.bert import BertModel, get_config
import dataset_jp_text as ds_jptxt
import dataset_IMDb as ds_imdb
from bert_cls import BertClassifier


def parse_arg():
    parser = argparse.ArgumentParser(description="Test BERT model.")
    parser.add_argument("--mecab_dict", type=str, help="MeCab dictionary.")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size.")
    parser.add_argument("--text_length", type=int, default=256, help="the length of texts.")
    #
    parser.add_argument("--IMDb", action="store_true", help="add this option when using IMDb dataset.")
    #
    parser.add_argument("conf", type=str, nargs=1, help="a BERT configuration file.")
    parser.add_argument("load_path", type=str, nargs=1, help="a path to trained net.")
    #
    parser.add_argument("tsv_file", type=str, nargs=1, help="TSV file for test.")
    parser.add_argument("vocab_file", type=str, nargs=1, help="a vocabulary file.")
    return parser.parse_args()


def test_model(net, data_loader, criterion, device):
    # epochの正解数を記録する変数
    epoch_corrects = 0

    for batch in tqdm(data_loader):  # testデータのDataLoader
        # batchはTextとLableの辞書オブジェクト
        # GPUが使えるならGPUにデータを送る
        inputs = batch.Text[0].to(device)  # 文章
        labels = batch.Label.to(device)  # ラベル

        # 順伝搬（forward）計算
        with torch.set_grad_enabled(False):
            # BertForIMDbに入力
            outputs = net(
                inputs,
                token_type_ids=None,
                attention_mask=None,
                output_all_encoded_layers=False,
                attention_show_flg=False)

            loss = criterion(outputs, labels)  # 損失を計算
            _, preds = torch.max(outputs, 1)  # ラベルを予測
            epoch_corrects += torch.sum(preds == labels.data)  # 正解数の合計を更新

    # 正解率
    epoch_acc = epoch_corrects.double() / len(data_loader.dataset)
    return epoch_acc


def run_main():
    args = parse_arg()
    if args.IMDb:
        ds = ds_imdb
    else:
        ds = ds_jptxt

    print("1. preparing datasets ... ", end="", flush=True)
    dataset_generator = ds.DataSetGenerator(args.vocab_file[0], args.text_length, mecab_dict=args.mecab_dict)
    test_ds = dataset_generator.loadTSV(args.tsv_file[0])
    test_dl = ds.get_data_loader(test_ds, args.batch_size, for_train=False)
    dataset_generator.build_vocab(test_ds)
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

    criterion = torch.nn.CrossEntropyLoss()  # クラス分けの場合
    # criterion = torch.nn.MSELoss()  # 数値予測の場合

    print("3. run tests. ", flush=True)
    epoch_acc = test_model(net, test_dl, criterion, device)
    print("# of test data: {} || Acc. {:.4f}".format(len(test_dl.dataset), epoch_acc))


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    run_main()
