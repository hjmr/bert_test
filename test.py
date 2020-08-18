import argparse
from tqdm import tqdm

import torch

# from dataset_jp_text import FieldSet, load_data_set, get_data_loader
from dataset_IMDb import FieldSet, load_data_set, get_data_loader


def parse_arg():
    parser = argparse.ArgumentParser(description="Test BERT model.")
    parser.add_argument("--mecab_dict", type=str, help="MeCab dictionary.")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size.")
    parser.add_argument("--text_length", type=int, default=256, help="the length of texts.")
    #
    parser.add_argument("--load_model", type=str, nargs=1, help="a path to trained net.")
    #
    parser.add_argument("test_tsv", type=str, nargs=1, help="TSV file for test data.")
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

    field_set = FieldSet(args.vocab_file[0], args.text_length, mecab_dict=args.mecab_dict)
    test_ds = load_data_set(args.test_tsv[0], field_set)
    test_dl = get_data_loader(test_ds, args.batch_size, for_train=False)

    net = torch.load(args.load_model[0], map_location=torch.device('cpu'))
    net.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    criterion = torch.nn.CrossEntropyLoss()  # クラス分けの場合
    # criterion = torch.nn.MSELoss()  # 数値予測の場合

    epoch_acc = test_model(net, test_dl, criterion, device)
    print("# of test data: {} || Acc. {:.4f}".format(len(test_dl.dataset), epoch_acc))


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    run_main()
