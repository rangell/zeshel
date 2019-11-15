STORAGE_BUCKET=gs://linking-data
BERT_BASE_DIR=${STORAGE_BUCKET}/uncased_L-12_H-768_A-12
EXPTS_DIR=${STORAGE_BUCKET}/tmp
TFRecords=${STORAGE_BUCKET}/tmp/TFRecords/mentions
USE_TPU=true
TPU_NAME=rangell

domain='test/star_trek'

EXP_NAME=BERT_fntn

python run_classifier.py \
  --do_train=false \
  --do_eval=true \
  --data_dir=$TFRecords \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --max_seq_length=256 \
  --num_cands=64 \
  --output_dir=$EXPTS_DIR/$EXP_NAME \
  --eval_domain=$domain \
  --use_tpu=$USE_TPU \
  --tpu_name=$TPU_NAME \
	--output_eval_file /tmp/eval.txt
