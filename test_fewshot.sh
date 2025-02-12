data_path="./test_datasets"

TASK=${1}
arch=${2}
sup_num=${3}
weight_path=${4}
results_path=${5}

n_gpu=1
batch_size=8
batch_size_valid=16
epoch=10
update_freq=1
lr=1e-4
MASTER_PORT=10092
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
seed=1

torchrun --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --user-dir ./unimol --train-subset train --valid-subset valid \
        --results-path $results_path \
        --num-workers 8 --ddp-backend=c10d \
        --task train_task --loss rank_softmax --arch $arch  \
        --max-pocket-atoms 256 \
        --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-8 --clip-norm 1.0 \
        --lr-scheduler polynomial_decay --lr $lr --max-epoch $epoch --batch-size $batch_size --batch-size-valid $batch_size_valid \
        --update-freq $update_freq --seed $seed \
        --log-interval 1 --log-format simple \
        --validate-interval 1 \
        --best-checkpoint-metric valid_mean_r2 --patience 100 --all-gather-list-size 2048000 \
        --no-save --save-dir $results_path --tmp-save-dir $results_path  \
        --find-unused-parameters \
        --maximize-best-checkpoint-metric \
        --split-method random --valid-set $TASK \
        --max-lignum 512 \
        --sup-num $sup_num \
        --restore-model $weight_path --few-shot true \
        --fp16 --fp16-init-scale 4 --fp16-scale-window 256