BATCH_SIZE=32
TEXT_LENGTH=256
EPOCH=2

MODEL_DIR=./models/ermk_model
CONF_FILE=${MODEL_DIR}/bert_config.json
BASE_MODEL=${MODEL_DIR}/pytorch_model.bin
VOCAB_FILE=${MODEL_DIR}/bert-base-uncased-vocab.txt

TRAIN_TSV=./data/IMDb/IMDb_train.tsv
SAVE_PATH=./results/IMDb/net_trained.pth
LOG_FILE=./results/IMDb/train.log


function run_once() {
    poetry run python train.py --IMDb --batch_size ${BATCH_SIZE} --text_length ${TEXT_LENGTH} --epoch ${EPOCH}  ${MECAB_OPT}  --save_path ${SAVE_PATH}  ${CONF_FILE}  ${BASE_MODEL}  ${TRAIN_TSV}  ${VOCAB_FILE} >& ${LOG_FILE} &
    tail -f ${LOG_FILE}
}

run_once
