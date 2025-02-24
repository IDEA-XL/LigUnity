num_cycles=${1}
begin_greedy=${2}
weight_path1=${3}
weight_path2=${4}
result_path1=${5}
result_path2=${6}

python ./active_learning_scripts/run_cycle_ours.py \
    --input_file ../PARank_data_curation/case_study/tyk2_fep_label.csv \
    --results_dir_1 ${result_path1} \
    --results_dir_2 ${result_path2} \
    --al_batch_size 100 \
    --num_cycles ${num_cycles} \
    --arch_1 pocket_ranking \
    --arch_2 protein_ranking \
    --weight_path_1 ${weight_path1} \
    --weight_path_2 ${weight_path2} \
    --lr 0.0001 \
    --device 0 \
    --master_port 10071 \
    --base_seed 42 \
    --begin_greedy ${begin_greedy}
