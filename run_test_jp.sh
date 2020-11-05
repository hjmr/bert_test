BATCH_SIZE=32
TEXT_LENGTH=256

MECAB_DICT_DIR=`mecab-config --dicdir`
MECAB_OPT=""
if [ -x ${MECAB_DICT_DIR}/mecab-ipadic-neologd ]; then
    MECAB_OPT="--mecab_dict ${MECAB_DICT_DIR}/mecab-ipadic-neologd"
fi

# remove comment if require Japanese text normalization.
# DO_NORMALIZE='--normalize_text'

MODEL_DIR=./models/Japanese_L-12_H-768_A-12_E-30_BPE
CONF_FILE=${MODEL_DIR}/bert_config.json
VOCAB_FILE=${MODEL_DIR}/vocab.txt

TRAINED_MODEL=./results/masuda/net_trained_10000.pth
TEST_TSV=./data/masuda/test_3000.tsv


function run_once() {
    poetry run python test.py --batch_size ${BATCH_SIZE} --text_length ${TEXT_LENGTH}  ${DO_NORMALIZE} ${MECAB_OPT}   ${CONF_FILE}  ${TRAINED_MODEL}  ${TEST_TSV}  ${VOCAB_FILE}
}

run_once
