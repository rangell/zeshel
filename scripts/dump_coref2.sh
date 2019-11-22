STORAGE_BUCKET=gs://linking-data
BERT_BASE_DIR=${STORAGE_BUCKET}/uncased_L-12_H-768_A-12
EXPTS_DIR=${STORAGE_BUCKET}/tmp
TFRecords=${STORAGE_BUCKET}/tmp/TFRecords/coref_mentions2
REPS_FILE=${EXPTS_DIR}/cereal/test2.pkl
USE_TPU=true
TPU_NAME=rangell

domain='test/test'

EXP_NAME=BERT_coref2
INIT=$BERT_BASE_DIR/bert_model.ckpt 

python3 run_coref2.py \
  --do_train=false \
  --do_predict=true \
  --data_dir=$TFRecords \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --max_seq_length=256 \
  --num_cands=16 \
  --predict_batch_size=8 \
  --output_dir=$EXPTS_DIR/$EXP_NAME \
  --eval_domain=$domain \
  --use_tpu=$USE_TPU \
  --tpu_name=$TPU_NAME \
  --output_rep_file=$REPS_FILE

