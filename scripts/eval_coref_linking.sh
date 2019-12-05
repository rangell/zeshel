STORAGE_BUCKET=gs://linking-data
BERT_BASE_DIR=${STORAGE_BUCKET}/uncased_L-12_H-768_A-12
EXPTS_DIR=${STORAGE_BUCKET}/tmp
TFRecords=${STORAGE_BUCKET}/tmp/TFRecords/mentions_coref_linking
USE_TPU=true
TPU_NAME=rangell-a

domain='val/val'

EXP_NAME=BERT_coref_linker

python run_coref_linking.py \
  --do_train=false \
  --do_predict=true \
  --data_dir=$TFRecords \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --max_seq_length=256 \
  --eval_batch_size=8 \
  --num_cands=64 \
  --num_coref=3 \
  --output_dir=$EXPTS_DIR/$EXP_NAME \
  --eval_domain=$domain \
  --use_tpu=$USE_TPU \
  --tpu_name=$TPU_NAME \
	--output_eval_file /tmp/eval.txt
