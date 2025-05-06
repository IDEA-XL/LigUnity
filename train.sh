data_path="./data"

save_root="./save"
save_name="screen_pocket"
save_dir="${save_root}/${save_name}/savedir_screen"
tmp_save_dir="${save_root}/${save_name}/tmp_save_dir_screen"
tsb_dir="${save_root}/${save_name}/tsb_dir_screen"
mkdir -p ${save_dir}
n_gpu=2
MASTER_PORT=10062
finetune_mol_model="./pretrain/mol_pre_no_h_220816.pt" # unimol pretrained mol model
finetune_pocket_model="./pretrain/pocket_pre_220816.pt" # unimol pretrained pocket model


batch_size=24
batch_size_valid=32
epoch=50
dropout=0.0
warmup=0.06
update_freq=1
dist_threshold=8.0
recycling=3
lr=1e-4

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --task train_task --loss rank_softmax --arch pocketscreen  \
       --max-pocket-atoms 256 \
       --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-8 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $batch_size --batch-size-valid $batch_size_valid \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --update-freq $update_freq --seed 1 \
       --tensorboard-logdir $tsb_dir \
       --log-interval 100 --log-format simple \
       --validate-interval 1 \
       --best-checkpoint-metric valid_bedroc --patience 2000 --all-gather-list-size 2048000 \
       --save-dir $save_dir --tmp-save-dir $tmp_save_dir --keep-best-checkpoints 8 --keep-last-epochs 10 \
       --find-unused-parameters \
       --maximize-best-checkpoint-metric \
       --finetune-pocket-model $finetune_pocket_model \
       --finetune-mol-model $finetune_mol_model \
       --valid-set CASF \
       --max-lignum 16 \
       --protein-similarity-thres 1.0 > ${save_root}/train_log/train_log_${save_name}.txt


save_name="screen_pocket_norank"
save_dir="${save_root}/${save_name}/savedir_screen"
tmp_save_dir="${save_root}/${save_name}/tmp_save_dir_screen"
tsb_dir="${save_root}/${save_name}/tsb_dir_screen"
mkdir -p ${save_dir}
n_gpu=2
MASTER_PORT=10062
finetune_mol_model="./pretrain/mol_pre_no_h_220816.pt" # unimol pretrained mol model
finetune_pocket_model="./pretrain/pocket_pre_220816.pt" # unimol pretrained pocket model

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --task train_task --loss rank_softmax --arch pocketscreen  \
       --max-pocket-atoms 256 \
       --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-8 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $batch_size --batch-size-valid $batch_size_valid \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --update-freq $update_freq --seed 1 \
       --tensorboard-logdir $tsb_dir \
       --log-interval 100 --log-format simple \
       --validate-interval 1 \
       --best-checkpoint-metric valid_bedroc --patience 2000 --all-gather-list-size 2048000 \
       --save-dir $save_dir --tmp-save-dir $tmp_save_dir --keep-best-checkpoints 8 --keep-last-epochs 10 \
       --find-unused-parameters \
       --maximize-best-checkpoint-metric \
       --finetune-pocket-model $finetune_pocket_model \
       --finetune-mol-model $finetune_mol_model \
       --valid-set CASF \
       --max-lignum 16 \
       --protein-similarity-thres 1.0 \
       --rank-weight 0.0 > ${save_root}/train_log/train_log_${save_name}.txt


save_name="screen_pocket_no_similar_protein0.8"
save_dir="${save_root}/${save_name}/savedir_screen"
tmp_save_dir="${save_root}/${save_name}/tmp_save_dir_screen"
tsb_dir="${save_root}/${save_name}/tsb_dir_screen"
mkdir -p ${save_dir}
n_gpu=2
MASTER_PORT=10062
finetune_mol_model="./pretrain/mol_pre_no_h_220816.pt" # unimol pretrained mol model
finetune_pocket_model="./pretrain/pocket_pre_220816.pt" # unimol pretrained pocket model

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --task train_task --loss rank_softmax --arch pocketscreen  \
       --max-pocket-atoms 256 \
       --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-8 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $batch_size --batch-size-valid $batch_size_valid \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --update-freq $update_freq --seed 1 \
       --tensorboard-logdir $tsb_dir \
       --log-interval 100 --log-format simple \
       --validate-interval 1 \
       --best-checkpoint-metric valid_bedroc --patience 2000 --all-gather-list-size 2048000 \
       --save-dir $save_dir --tmp-save-dir $tmp_save_dir --keep-best-checkpoints 8 --keep-last-epochs 10 \
       --find-unused-parameters \
       --maximize-best-checkpoint-metric \
       --finetune-pocket-model $finetune_pocket_model \
       --finetune-mol-model $finetune_mol_model \
       --valid-set CASF \
       --max-lignum 16 \
       --protein-similarity-thres 0.8 > ${save_root}/train_log/train_log_${save_name}.txt


save_name="screen_pocket_no_similar_protein"
save_dir="${save_root}/${save_name}/savedir_screen"
tmp_save_dir="${save_root}/${save_name}/tmp_save_dir_screen"
tsb_dir="${save_root}/${save_name}/tsb_dir_screen"
mkdir -p ${save_dir}
n_gpu=2
MASTER_PORT=10062
finetune_mol_model="./pretrain/mol_pre_no_h_220816.pt" # unimol pretrained mol model
finetune_pocket_model="./pretrain/pocket_pre_220816.pt" # unimol pretrained pocket model

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --task train_task --loss rank_softmax --arch pocketscreen  \
       --max-pocket-atoms 256 \
       --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-8 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $batch_size --batch-size-valid $batch_size_valid \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --update-freq $update_freq --seed 1 \
       --tensorboard-logdir $tsb_dir \
       --log-interval 100 --log-format simple \
       --validate-interval 1 \
       --best-checkpoint-metric valid_bedroc --patience 2000 --all-gather-list-size 2048000 \
       --save-dir $save_dir --tmp-save-dir $tmp_save_dir --keep-best-checkpoints 8 --keep-last-epochs 10 \
       --find-unused-parameters \
       --maximize-best-checkpoint-metric \
       --finetune-pocket-model $finetune_pocket_model \
       --finetune-mol-model $finetune_mol_model \
       --valid-set CASF \
       --max-lignum 16 \
       --protein-similarity-thres 0.4 > ${save_root}/train_log/train_log_${save_name}.txt