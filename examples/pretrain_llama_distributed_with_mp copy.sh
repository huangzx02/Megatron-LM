#!/bin/bash

# Runs the "7B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=9302
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
TP=2

CHECKPOINT_PATH=/workspace/checkpoints
VOCAB_FILE=/workspace/megatron/gpt2-vocab.json
MERGE_FILE=/workspace/megatron/gpt2-merges.txt
DATA_PATH=/workspace/dataset/gpt2_text_document

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --sequence-parallel \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 4 \
    --global-batch-size 16 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

LLAMA_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size 1 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --load ${CHECKPOINT_DIR} \
    --exit-on-missing-checkpoint \
    --use-checkpoint-args \
    --no-load-optim \
    --no-load-rng \
    --fp16 \
    --untie-embeddings-and-output-weights \
    --use-rotary-position-embeddings \
    --normalization RMSNorm \
    --no-position-embedding \
    --no-masked-softmax-fusion \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 8,1,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 1
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $LLAMA_ARGS_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH

