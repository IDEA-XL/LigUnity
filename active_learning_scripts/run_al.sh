num_cycles=${1}
begin_greedy=${2}

python ./active_learning_scripts/run_cycle_ours.py \
    --input_file ../PARank_data_curation/case_study/tyk2_fep_label.csv \
    --results_dir_1 "/cto_studio/xtalpi_lab/fengbin/PARank_save/FEP_pocket_epoch20_lr1/test_result/TYK2_FEP_AL_halfgreedy_fusion_pocket_debug" \
    --results_dir_2 "/cto_studio/xtalpi_lab/fengbin/PARank_save/FEP_pocket_epoch20_lr1/test_result/TYK2_FEP_AL_halfgreedy_fusion_seq_debug" \
    --al_batch_size 100 \
    --num_cycles ${num_cycles} \
    --arch_1 pocket_ranking \
    --arch_2 protein_ranking \
    --weight_path_1 "/cto_studio/xtalpi_lab/fengbin/PARank_save/FEP_pocket_epoch20_lr1/savedir_screen_con0.5_1/checkpoint_avg_41-50.pt" \
    --weight_path_2 "/cto_studio/xtalpi_lab/fengbin/PARank_save/FEP_seq_epoch20_lr1/savedir_screen_con0.5_1/checkpoint_avg_41-50.pt" \
    --lr 0.0001 \
    --device 0 \
    --master_port 10071 \
    --base_seed 42 \
    --begin_greedy ${begin_greedy}