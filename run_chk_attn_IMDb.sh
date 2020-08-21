BATCH_SIZE=32
TEXT_LENGTH=256

MODEL_DIR=./models/ermk_model
CONF_FILE=${MODEL_DIR}/bert_config.json
VOCAB_FILE=${MODEL_DIR}/bert-base-uncased-vocab.txt

TRAINED_MODEL=./results/IMDb/net_trained.pth
TSV_FILE=./data/IMDb/IMDb_train.tsv


INDEX=0
HTML_FILE=./results/IMDb/attention/att_${INDEX}.html


function run_once() {
    poetry run python check_attention.py --IMDb --batch_size ${BATCH_SIZE} --text_length ${TEXT_LENGTH} --index ${INDEX} --save_html ${HTML_FILE}  ${CONF_FILE}  ${TRAINED_MODEL}  ${TSV_FILE}  ${VOCAB_FILE}
}

run_once
