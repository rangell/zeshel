STORAGE_BUCKET=gs://linking-data
BERT_BASE_DIR=${STORAGE_BUCKET}/uncased_L-12_H-768_A-12
EXPTS_DIR=${STORAGE_BUCKET}/tmp
TFRecords=${EXPTS_DIR}/TFRecords/cand_gen
REPS_FILE=${EXPTS_DIR}/cand_gen_reps/train.pkl
USE_TPU=true
TPU_NAME=rangell

domain='train/train'

EXP_NAME=BERT_cand_gen
INIT=$BERT_BASE_DIR/bert_model.ckpt 

python3 run_cand_gen.py \
  --do_train=false \
  --do_predict=true \
  --data_dir=$TFRecords \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --max_seq_length=128 \
  --num_cands=64 \
  --predict_batch_size=8 \
  --output_dir=$EXPTS_DIR/$EXP_NAME \
  --eval_domain=$domain \
  --use_tpu=$USE_TPU \
  --tpu_name=$TPU_NAME \
  --output_rep_file=$REPS_FILE

