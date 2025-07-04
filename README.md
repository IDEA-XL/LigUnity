## General
This repository contains the code for **LigUnity**: **A foundation model for protein-ligand affinity prediction through jointly optimizing virtual screening and hit-to-lead optimization.**

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green?style=flat-square)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red?style=flat-square)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)
[![DOI:10.1101/2021.01.08.425840](http://img.shields.io/badge/DOI-10.1101/2025.02.17.638554-B31B1B.svg)](https://doi.org/10.1101/2025.02.17.638554) 
[![GitHub Link](https://img.shields.io/badge/GitHub-blue?style=flat-square&logo=github)](https://github.com/IDEA-XL/LigUnity)

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

Please first download checkpoints and processed dataset before running
- Download our procesed Dekois 2.0 dataset from https://doi.org/10.6084/m9.figshare.27967422
- Download LIT-PCBA and DUD-E datasets from https://drive.google.com/drive/folders/1zW1MGpgunynFxTKXC2Q4RgWxZmg6CInV?usp=sharing
- Clone model checkpoint from https://huggingface.co/fengb/LigUnity_VS (test proteins in DUD-E, Dekois, and LIT-PCBA are removed from the training set)
- Clone dataset from https://figshare.com/articles/dataset/LigUnity_project_data/27966819 and unzip them all (you can ignore .lmdb file if you only want to reproduce test result).

```
# run pocket/protein and ligand encoder model
path2weight="absolute path to the checkpoint of pocket_ranking"
CUDA_VISIBLE_DEVICES=0 bash test.sh ALL pocket_ranking ${path2weight} "./result/pocket_ranking"
CUDA_VISIBLE_DEVICES=0 bash test.sh BDB pocket_ranking ${path2weight} "./result/pocket_ranking"
CUDA_VISIBLE_DEVICES=0 bash test.sh PDB pocket_ranking ${path2weight} "./result/pocket_ranking"

path2weight="absolute path to the checkpoint of protein_ranking"
CUDA_VISIBLE_DEVICES=0 bash test.sh ALL protein_ranking ${path2weight} "./result/protein_ranking"
CUDA_VISIBLE_DEVICES=0 bash test.sh BDB protein_ranking ${path2weight} "./result/protein_ranking"
CUDA_VISIBLE_DEVICES=0 bash test.sh PDB protein_ranking ${path2weight} "./result/protein_ranking"

# train H-GNN model
cd ./HGNN
python main.py --data_root ${path2data} --result_root "../result/pocket_ranking"
python main.py --data_root ${path2data} --result_root "../result/protein_ranking"

# get final prediction of our model
python ensemble_result.py DUDE PCBA DEKOIS
```


### Reproduce results on FEP benchmarks (zero-shot)

Please first download checkpoints before running
- Clone model checkpoint from https://huggingface.co/fengb/LigUnity_pocket_ranking and https://huggingface.co/fengb/LigUnity_protein_ranking (test ligands and assays in FEP benchmarks are removed from the training set)

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
# use the same checkpoints as in zero-shot
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
# use the same checkpoints as in FEP experiments
path1="path to checkpoint of pocket_ranking"
path2="path to checkpoint of protein_ranking"
result1="./result/pocket_ranking/TYK2"
result2="./result/protein_ranking/TYK2"

# run active learning cycle for 5 iters with pure greedy strategy
bash ./active_learning_scripts/run_al.sh 5 0 path1 path2 result1 result2
```
## Citation

```
@article{feng2025foundation,
  title={A foundation model for protein-ligand affinity prediction through jointly optimizing virtual screening and hit-to-lead optimization},
  author={Feng, Bin and Liu, Zijing and Yang, Mingjun and Zou, Junjie and Cao, He and Li, Yu and Zhang, Lei and Wang, Sheng},
  journal={bioRxiv},
  pages={2025--02},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}

@article{feng2024bioactivity,
  title={A bioactivity foundation model using pairwise meta-learning},
  author={Feng, Bin and Liu, Zequn and Huang, Nanlan and Xiao, Zhiping and Zhang, Haomiao and Mirzoyan, Srbuhi and Xu, Hanwen and Hao, Jiaran and Xu, Yinghui and Zhang, Ming and others},
  journal={Nature Machine Intelligence},
  volume={6},
  number={8},
  pages={962--974},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

## Acknowledgments 

This project was built based on Uni-Mol (https://github.com/deepmodeling/Uni-Mol)

Parts of our code reference the implementation from DrugCLIP (https://github.com/bowen-gao/DrugCLIP) by bowen-gao
