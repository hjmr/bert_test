from torch import nn


class BertClassifier(nn.Module):
    def __init__(self, bert_base, out_features):
        super(BertClassifier, self).__init__()

        self.bert = bert_base
        self.cls = nn.Linear(
            in_features=self.bert.pooler.dense.out_features,
            out_features=out_features)

        nn.init.normal_(self.cls.weight, std=0.02)
        nn.init.normal_(self.cls.bias, 0)
        self._setup_grad()

    def _setup_grad(self):
        # 1. まず全部を、勾配計算Falseにしてしまう
        for _, param in self.named_parameters():
            param.requires_grad = False

        # 2. 最後のBertLayerモジュールを勾配計算ありに変更
        for _, param in self.bert.encoder.layer[-1].named_parameters():
            param.requires_grad = True

        # 3. 識別器を勾配計算ありに変更
        for _, param in self.cls.named_parameters():
            param.requires_grad = True

    def forward(
            self,
            input_ids,
            token_type_ids=None,
            attention_mask=None,
            output_all_encoded_layers=False,
            attention_show_flg=False):
        '''
        input_ids: [batch_size, sequence_length]の文章の単語IDの羅列
        token_type_ids: [batch_size, sequence_length]の、各単語が1文目なのか、2文目なのかを示すid
        attention_mask: Transformerのマスクと同じ働きのマスキングです
        output_all_encoded_layers: 最終出力に12段のTransformerの全部をリストで返すか、最後だけかを指定
        attention_show_flg: Self-Attentionの重みを返すかのフラグ
        '''

        # BERTの基本モデル部分の順伝搬
        # 順伝搬させる
        if attention_show_flg == True:
            encoded_layers, _, attention_probs = self.bert(
                input_ids, token_type_ids, attention_mask, output_all_encoded_layers, attention_show_flg)
        elif attention_show_flg == False:
            encoded_layers, _ = self.bert(
                input_ids, token_type_ids, attention_mask, output_all_encoded_layers, attention_show_flg)
        # 入力文章の1単語目[CLS]の特徴量を使用して、ポジ・ネガを分類します
        vec_0 = encoded_layers[:, 0, :]
        vec_0 = vec_0.view(-1, self.cls.in_features)  # sizeを[batch_size, hidden_sizeに変換
        out = self.cls(vec_0)

        # attention_showのときは、attention_probs（1番最後の）もリターンする
        if attention_show_flg == True:
            return out, attention_probs
        elif attention_show_flg == False:
            return out


if __name__ == "__main__":
    import argparse
    from utils.bert import BertModel, get_config, set_learned_params

    def parse_arg():
        parser = argparse.ArgumentParser(description="Test for BERT for Japanese Texts.")
        parser.add_argument("conf", type=str, nargs=1, help="a configuration file.")
        parser.add_argument("model", type=str, nargs=1, help="a trained model file.")
        return parser.parse_args()

    args = parse_arg()
    conf = get_config(file_path=args.conf[0])
    bert_base = BertModel(conf)
    bert_base = set_learned_params(bert_base, weights_path=args.model[0])
    net = BertClassifier(bert_base, out_features=2)
    print(net)
