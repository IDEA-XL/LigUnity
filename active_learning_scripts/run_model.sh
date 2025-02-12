data_path="./test_datasets"

n_gpu=1

batch_size=1
batch_size_valid=1
epoch=20
update_freq=1
#lr=1e-3
#MASTER_PORT=10075
#arch=pocket_ranking

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

arch=${1} # model architecture
weight_path=${2} # path for pretrained model
results_path=${3} #
result_file=${4} #
lr=${5} # learning rate
MASTER_PORT=${6}
train_ligf=${7} # !! input path for training ligands file (.csv format)
test_ligf=${8} # !! input path for test ligands file (.csv format)
device=${9} # cuda device

if [[ "$arch" == "pocketregression" ]] || [[ "$arch" == "DTA" ]]; then
    loss="mseloss"
else
    loss="rank_softmax"
fi


CUDA_VISIBLE_DEVICES=${device} python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --user-dir ./unimol --train-subset train --valid-subset valid \
     --results-path $results_path \
     --num-workers 8 --ddp-backend=c10d \
     --task train_task --loss ${loss} --arch $arch  \
     --max-pocket-atoms 256 \
     --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-8 --clip-norm 1.0 \
     --lr-scheduler polynomial_decay --lr $lr --max-epoch $epoch --batch-size $batch_size --batch-size-valid $batch_size_valid \
     --update-freq $update_freq --seed 1 \
     --log-interval 1 --log-format simple \
     --validate-interval 1 --validate-begin-epoch 15 \
     --best-checkpoint-metric valid_mean_r2 --patience 100 --all-gather-list-size 2048000 \
     --no-save --save-dir $results_path --tmp-save-dir $results_path  \
     --find-unused-parameters \
     --maximize-best-checkpoint-metric \
     --valid-set TYK2 \
     --max-lignum 512 --test-max-lignum 10000 \
     --restore-model $weight_path --few-shot true \
     --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
     --active-learning-resfile ${result_file} \
     --case-train-ligfile ${train_ligf} --case-test-ligfile ${test_ligf}

