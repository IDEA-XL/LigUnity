## General
This repository contains the code for **LigUnity**: A foundation model for protein-ligand affinity prediction through jointly optimizing virtual screening and hit-to-lead optimization.

## Instruction on running our model

### Direct inference
Colab demo for code inference with given protein and unmeasured ligands.

https://colab.research.google.com/drive/11Fx6mO51rRkPvq71qupuUmscfBw8Dw5R?usp=sharing

### Few-shot fine-tuning
Colab demo for few-shot fine-tuning with given protein, few measure ligands for fine-tuning and unmeasured ligands for testing.

https://colab.research.google.com/drive/1gf0HhgyqI4qBjUAUICCvDa-FnTaARmR_?usp=sharing

Please feel free to contact me by email if there is any problem with the code or paper: fengbin@idea.edu.cn.

## Abstract

Protein-ligand binding affinity plays an important role in drug discovery, especially during virtual screening and hit-to-lead optimization. Computational chemistry and machine learning methods have been developed to investigate these tasks. Despite the encouraging performance, virtual screening and hit-to-lead optimization are often studied separately by existing methods, partially because they are performed sequentially in the existing drug discovery pipeline, thereby overlooking their interdependency and complementarity. To address this problem, we propose LigUnity, a foundation model for protein-ligand binding prediction by jointly optimizing virtual screening and hit-to-lead optimization. 
In particular, LigUnity learns coarse-grained active/inactive distinction for virtual screening, and fine-grained pocket-specific ligand preference for hit-to-lead optimization. 
We demonstrate the effectiveness and versatility of LigUnity on eight benchmarks across virtual screening and hit-to-lead optimization. In virtual screening, LigUnity outperforms 24 competing methods with more than 50% improvement on the DUD-E and Dekois 2.0 benchmarks, and shows robust generalization to novel proteins. In hit-to-lead optimization, LigUnity achieves the best performance on split-by-time, split-by-scaffold, and split-by-unit settings, further demonstrating its potential as a cost-effective alternative to free energy perturbation (FEP) calculations. We further showcase how LigUnity can be employed in an active learning framework to efficiently identify active ligands for TYK2, a therapeutic target for autoimmune diseases, yielding over 40% improved prediction performance. Collectively, these comprehensive results establish LigUnity as a versatile foundation model for both virtual screening and hit-to-lead optimization, offering broad applicability across the drug discovery pipeline through accurate protein-ligand affinity predictions.



## Reproduce results in our paper

### Reproduce results on virtual screening benchmarks

Please first download the processed datased before running
- Download our procesed Dekois 2.0 dataset from https://doi.org/10.6084/m9.figshare.27967422
- Download LIT-PCBA and DUD-E datasets from https://drive.google.com/drive/folders/1zW1MGpgunynFxTKXC2Q4RgWxZmg6CInV?usp=sharing

```
# run pocket/protein and ligand encoder model
path2weight="path to checkpoint of pocket_ranking"
path2result="./result/pocket_ranking"
CUDA_VISIBLE_DEVICES=0 bash test.sh ALL pocket_ranking ${path2weight} ${path2result}

path2weight="path to checkpoint of protein_ranking"
path2result="./result/protein_ranking"
CUDA_VISIBLE_DEVICES=0 bash test.sh ALL protein_ranking ${path2weight} ${path2result}

# run H-GNN model
# coming soon

# get final prediction of our model
python ensemble_result.py DUDE PCBA DEKOIS
```


### Reproduce results on FEP benchmarks (zero-shot)
```
# run pocket/protein and ligand encoder model
for r in {1..6} do
    path2weight="path to checkpoint of pocket_ranking"
    path2result="./result/pocket_ranking/FEP/repeat_{r}"
    CUDA_VISIBLE_DEVICES=0 bash test.sh FEP pocket_ranking ${path2weight} ${path2result}
    
    path2weight="path to checkpoint of protein_ranking"
    path2result="./result/protein_ranking/FEP/repeat_{r}"
    CUDA_VISIBLE_DEVICES=0 bash test.sh FEP protein_ranking ${path2weight} ${path2result}
done

# get final prediction of our model
python ensemble_result.py FEP
```

### Reproduce results on FEP benchmarks (few-shot)
```
# run few-shot fine-tuning
for r in {1..6} do
    path2weight="path to checkpoint of pocket_ranking"
    path2result="./result/pocket_ranking/FEP_fewshot/repeat_{r}"
    support_num=0.6
    CUDA_VISIBLE_DEVICES=0 bash test_fewshot.sh FEP pocket_ranking support_num ${path2weight} ${path2result}
    
    path2weight="path to checkpoint of protein_ranking"
    path2result="./result/protein_ranking/FEP_fewshot/repeat_{r}"
    CUDA_VISIBLE_DEVICES=0 bash test_fewshot.sh FEP protein_ranking support_num ${path2weight} ${path2result}
done

# get final prediction of our model
python ensemble_result_fewshot.py FEP_fewshot support_num
```

### Reproduce results on active learning
to speed up the active learning process, you should modify the unicore code 
1. find the installed dir of unicore (root-to-unicore)
```
python -c "import unicore; print('/'.join(unicore.__file__.split('/')[:-2]))"
```

2. goto root-to-unicore/unicore/options.py line 250, add following line
```
    group.add_argument('--validate-begin-epoch', type=int, default=0, metavar='N',
                        help='validate begin epoch')
```

3. goto root-to-unicore/unicore_cli/train.py line 303, add one line
```
    do_validate = (
        (not end_of_epoch and do_save)
        or (
            end_of_epoch
            and epoch_itr.epoch >= args.validate_begin_epoch # !!!! add this line
            and epoch_itr.epoch % args.validate_interval == 0
            and not args.no_epoch_checkpoints
        )
        or should_stop
        or (
            args.validate_interval_updates > 0
            and num_updates > 0
            and num_updates % args.validate_interval_updates == 0
        )
    ) and not args.disable_validation
```

4. run the active learning procedure
```
path1="path to checkpoint of pocket_ranking"
path2="path to checkpoint of protein_ranking"
result1="./result/pocket_ranking/TYK2"
result2="./result/protein_ranking/TYK2"

# run active learning cycle for 5 iters with pure greedy strategy
bash ./active_learning_scripts/run_al.sh 5 0 path1 path2 result1 result2
```

## Acknowledgments 

This project was built based on Uni-Mol (https://github.com/deepmodeling/Uni-Mol)

Parts of our code reference the implementation from DrugCLIP (https://github.com/bowen-gao/DrugCLIP) by bowen-gao
