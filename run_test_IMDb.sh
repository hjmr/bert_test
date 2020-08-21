BATCH_SIZE=32
TEXT_LENGTH=256

MODEL_DIR=./models/ermk_model
CONF_FILE=${MODEL_DIR}/bert_config.json
VOCAB_FILE=${MODEL_DIR}/bert-base-uncased-vocab.txt

TRAINED_MODEL=./results/IMDb/net_trained.pth
TEST_TSV=./data/IMDb/IMDb_test.tsv


function run_once() {
    poetry run python test.py --IMDb --batch_size ${BATCH_SIZE} --text_length ${TEXT_LENGTH}   ${CONF_FILE}  ${TRAINED_MODEL}  ${TEST_TSV}  ${VOCAB_FILE}
}

run_once
