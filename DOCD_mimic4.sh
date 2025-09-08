export DATA_DIR='data/mimic4-3.0'
export CUDA_VISIBLE_DEVICES="2"

LOSS_COEF=0.1
MODEL_NAME='DOCD'
DATASET_NAME='mimiciv'
HIDDEN_SIZE=256
INTERMEDIATE_SIZE=$((HIDDEN_SIZE *4))
MAX_STEPS=100000
NUM_STACKS=8

for LR in 1e-2 1e-3 1e-4; do
  for DROPOUT in 0.2 0.3 0.4; do
      OUTPUT_DIR="output/${DATASET_NAME}${MAX_STEPS}steps_${HIDDEN_SIZE}/${LR}_${DROPOUT}"
      mkdir -p $OUTPUT_DIR
      python train.py \
          --train_percentage 1\
          --dataset_name $DATASET_NAME \
          --model_name $MODEL_NAME \
          --data_dir $DATA_DIR \
          --fold 50 \
          --batch_size 128 \
          --output_dir $OUTPUT_DIR \
          --hidden_size $HIDDEN_SIZE \
          --intermediate_size $INTERMEDIATE_SIZE \
          --output_hidden_states \
          --output_attentions \
          --do_train \
          --do_eval \
          --do_test \
          --output_model \
          --max_steps $MAX_STEPS \
          --hidden_dropout_prob $DROPOUT \
          --num_stacks $NUM_STACKS \
          --learning_rate $LR \
          --loss_coef $LOSS_COEF\
          --use_prior \
          --use_guide
  done
done

