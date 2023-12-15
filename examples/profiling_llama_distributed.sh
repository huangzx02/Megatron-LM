#!/bin/bash

export NCCL_IB_HCA=mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_8:1,mlx5_9:1,mlx5_10:1
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=2
export NCCL_IB_RETRY_CNT=7
export NCCL_P2P_DISABLE=0
export NCCL_CROSS_NIC=0

export OMP_NUM_THREADS=4
export CUDA_DEVICE_MAX_CONNECTIONS=1  #PP


GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=9302
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# Parallel config
TP=8
PP=1

# Data config
DP_SIZE=$((WORLD_SIZE/TP/PP))
BATCH_SIZE=1
ACCUMULATION_STEPS=8
GLOBAL_BATCH_SIZE=$(($BATCH_SIZE*$DP_SIZE*$ACCUMULATION_STEPS))

# Model config
SEQ_LENGTH=4096

# Original
# HIDDEN_SIZE=1024
# FFN_HIDDEN_SIZE=4096
# NUM_LAYERS=16
# NUM_HEADS=16
# NUM_ATTN_HEADS=16

# LLaMa 7B
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=11008
NUM_LAYERS=32
NUM_HEADS=32
NUM_ATTN_HEADS=32

# LLaMa 13B
# HIDDEN_SIZE=5120
# FFN_HIDDEN_SIZE=13824
# NUM_LAYERS=40
# NUM_HEADS=40
# NUM_ATTN_HEADS=40

DATA_PATH=/workspace/dataset/gpt2_text_document
TRAIN_SAMPLES=10000
TRAIN_ITERS=$(($TRAIN_SAMPLES/$GLOBAL_BATCH_SIZE))

VPP_SIZE=$(($NUM_LAYERS/$TP))

EXP_NAME=profiling-llama
EXP_FOLDER=/workspace/megatron
CHECKPOINT_PATH=$EXP_FOLDER/checkpoints
TENSORBOARD_PATH=$EXP_FOLDER/tensorboard

mkdir -p $CHECKPOINT_PATH $TENSORBOARD_PATH

TOKENIZER_PATH=/workspace/megatron/llama2_tokenizer.model
TOKENIZER_TYPE=Llama2Tokenizer

SEED=42

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT"

GPT_ARGS="
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --sequence-parallel \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTN_HEADS \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $SEQ_LENGTH \
    --position-embedding-type rope \
    --micro-batch-size $BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --make-vocab-size-divisible-by $TP \
    --lr 2e-4 \
    --lr-decay-iters $TRAIN_ITERS \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.003 \
    --min-lr 1.5e-4 \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --weight-decay 1e-2 \
    --swiglu \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --clip-grad 1.0 \
    --bf16 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 2 \
    --no-check-for-nan-in-loss-and-grad"

DATA_ARGS="
    --data-path $DATA_PATH \
    --train-iters $TRAIN_ITERS \
    --tokenizer-type $TOKENIZER_TYPE \
    --tokenizer-model $TOKENIZER_PATH \
    --dataloader-type cyclic \
    --num-workers 4 \
    --split 1000,0,0"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 100 \
    --eval-interval 1000 \
    --eval-iters 0"

torchrun $DISTRIBUTED_ARGS pretrain_llama.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --seed $SEED \
    --distributed-backend nccl \
    --use-flash-attn \
    --save $CHECKPOINT_PATH \
    --use-distributed-optimizer \
    --overlap-param-gather \
    --empty-unused-memory-level 2 \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-log-interval 1 \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-memory-to-tensorboard