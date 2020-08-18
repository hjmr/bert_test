import argparse
import random
import time

import numpy as np

import torch

from utils.bert import BertModel, get_config, set_learned_params
from dataset_jp_text import FieldSet, load_data_set, get_data_loader
# from dataset_IMDb import FieldSet, load_data_set, get_data_loader
from bert_cls import BertClassifier


def parse_arg():
    parser = argparse.ArgumentParser(description="BERT for Japanese Texts.")
    parser.add_argument("--mecab_dict", type=str, help="MeCab dictionary.")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size.")
    parser.add_argument("--text_length", type=int, default=256, help="the length of texts.")
    parser.add_argument("--random_seed", type=int, help="a random seed.")
    #
    parser.add_argument("--conf", type=str, nargs=1, help="a BERT configuration file.")
    parser.add_argument("--model", type=str, nargs=1, help="a trained BERT model file.")
    #
    parser.add_argument("--epoch", type=int, default=10, help="train epochs.")
    parser.add_argument("--save_path", type=str, help="a file to save trained net.")
    #
    parser.add_argument("train_tsv", type=str, nargs=1, help="TSV file for train data.")
    parser.add_argument("vocab_file", type=str, nargs=1, help="a vocabulary file.")
    return parser.parse_args()


def init_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_network(net):
    # 1. まず全部を、勾配計算Falseにしてしまう
    for _, param in net.named_parameters():
        param.requires_grad = False

    # 2. 最後のBertLayerモジュールを勾配計算ありに変更
    for _, param in net.bert.encoder.layer[-1].named_parameters():
        param.requires_grad = True

    # 3. 識別器を勾配計算ありに変更
    for _, param in net.cls.named_parameters():
        param.requires_grad = True


def get_optimizer(net):
    optimizer = torch.optim.Adam([
        {'params': net.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
        {'params': net.cls.parameters(), 'lr': 5e-5}
    ], betas=(0.9, 0.999))
    return optimizer


def train_iter_log(iteration, loss, duration, acc):
    print('Iter {} || Loss: {:.4f} Acc: {} || {:.4f} sec'.format(
        iteration, loss.item(), acc, duration))


def train_epoch_log(epoch, num_epochs, phase, epoch_loss, epoch_acc):
    print('Epoch {}/{} | {:^5} | Loss: {:.4f} Acc: {:.4f}'.format(
        epoch+1, num_epochs, phase, epoch_loss, epoch_acc))


def train_model(net, data_loader_set, criterion, optimizer, num_epochs):

    # GPUが使える場合はGPUで計算
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # ミニバッチのサイズ
    batch_size = data_loader_set["train"].batch_size

    # epochのループ
    for epoch in range(num_epochs):
        # epochごとの訓練と検証のループ
        for phase in ["train", "validation"]:
            if phase == "train":
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数
            iteration = 1

            # 開始時刻を保存
            t_epoch_start = time.time()
            t_iter_start = time.time()

            # データローダーからミニバッチを取り出すループ
            for batch in (data_loader_set[phase]):
                # batchはTextとLabelの辞書型変数

                # GPUが使えるならGPUにデータを送る
                inputs = batch.Text[0].to(device)  # 文章
                labels = batch.Label.to(device)  # ラベル

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(
                        inputs,
                        token_type_ids=None,
                        attention_mask=None,
                        output_all_encoded_layers=False,
                        attention_show_flg=False)

                    loss = criterion(outputs, labels)  # 損失を計算
                    _, preds = torch.max(outputs, 1)  # ラベルを予測

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if (iteration % 10 == 0):  # 10iterに1度、lossを表示
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            acc = (torch.sum(preds == labels.data)).double()/batch_size
                            train_iter_log(iteration, loss.item(), duration, acc)
                            t_iter_start = time.time()

                    iteration += 1

                    # 損失と正解数の合計を更新
                    epoch_loss += loss.item() * batch_size
                    epoch_corrects += torch.sum(preds == labels.data)

            # epochごとのlossと正解率
            t_epoch_finish = time.time()
            epoch_loss = epoch_loss / len(data_loader_set[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(data_loader_set[phase].dataset)

            train_epoch_log(epoch, num_epochs, phase, epoch_loss, epoch_acc)
            t_epoch_start = time.time()
    return net


def run_main():
    args = parse_arg()
    if args.random_seed is not None:
        init_random_seed(args.random_seed)

    field_set = FieldSet(args.vocab_file[0], args.mecab_dict, args.text_length)

    data_set = load_data_set(args.train_tsv[0], field_set)
    train_ds, validation_ds = data_set.split(split_ratio=0.8)
    field_set.build_vocab(train_ds)

    train_dl = get_data_loader(train_ds, args.batch_size, for_train=True)
    validation_dl = get_data_loader(validation_ds, args.batch_size, for_train=False)

    conf = get_config(file_path=args.conf[0])
    bert_base = BertModel(conf)
    bert_base = set_learned_params(bert_base, weights_path=args.model[0])
    net = BertClassifier(bert_base, out_features=1)

    optimizer = get_optimizer(net)
    criterion = torch.nn.CrossEntropyLoss()  # クラス分けの場合
    # criterion = torch.nn.MSELoss()  # 数値予測の場合

    data_loader_set = {"train": train_dl, "validation": validation_dl}
    net_trained = train_model(net, data_loader_set, criterion, optimizer, args.epoch)
    if args.save_path is not None:
        torch.save(net_trained, args.save_path)


if __name__ == "__main__":
    run_main()
