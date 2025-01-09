import json
import os.path

import math
from functools import lru_cache

import torch
from unicore.data import UnicoreDataset
import numpy as np
from . import data_utils
import rdkit
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from multiprocessing import Pool
from tqdm import tqdm

def get_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp_numpy = np.zeros((0,), np.int8)  # Generate target pointer to fill
    if mol is None:
        return None
    fingerprints_vect = rdFingerprintGenerator.GetCountFPs(
        [mol], fpType=rdFingerprintGenerator.MorganFP
    )[0]
    DataStructs.ConvertToNumpyArray(fingerprints_vect, fp_numpy)
    return fp_numpy

class PairDataset(UnicoreDataset):
    def __init__(self, args, pocket_dataset, mol_dataset, labels, split, use_cache=True, cache_dir=None):
        self.args = args
        self.pocket_dataset = pocket_dataset
        self.mol_dataset = mol_dataset
        self.labels = labels

        # use the cached file, or it will take loooooong time to load
        if use_cache:
            pocket_name2idx_file = f"{cache_dir}/cache/pocket_name2idx_train_blend.json"
            if os.path.exists(pocket_name2idx_file):
                self.pocket_name2idx = json.load(open(pocket_name2idx_file))
            else:
                self.pocket_name2idx = {x["pocket_name"]:i for i,x in enumerate(self.pocket_dataset)}
                json.dump(self.pocket_name2idx, open(pocket_name2idx_file, "w"))
        else:
            self.pocket_name2idx = {x["pocket_name"]: i for i, x in enumerate(self.pocket_dataset)}

        if use_cache:
            mol_smi2idx_file = f"{cache_dir}/cache/mol_smi2idx_train_blend.json"
            if os.path.exists(mol_smi2idx_file):
                self.mol_smi2idx = json.load(open(mol_smi2idx_file))
            else:
                self.mol_smi2idx = {x["smi_name"]: i for i, x in enumerate(self.mol_dataset)}
                json.dump(self.mol_smi2idx, open(mol_smi2idx_file, "w"))
        else:
            self.mol_smi2idx = {x["smi_name"]: i for i, x in enumerate(self.mol_dataset)}

        uniprot_ids = [x["uniprot"] for x in labels]
        self.uniprot_id_dict = {x:i for i,x in enumerate(set(uniprot_ids))}
        self.split = split
        if self.split == "train":
            self.max_lignum = args.max_lignum # default=16
        else:
            self.max_lignum = args.test_max_lignum # default 512

        if self.split == "train":
            trainidxmap = []
            for idx, assay_item in enumerate(self.labels):
                lig_info = assay_item["ligands"]
                trainidxmap += [idx]*math.ceil(len(lig_info)/max(self.max_lignum, 32))
            self.trainidxmap = trainidxmap

        self.epoch = 0


    def __len__(self):
        if self.split == "train":
            import os
            world_size = int(os.environ["WORLD_SIZE"])
            div = self.args.batch_size * world_size
            return (len(self.trainidxmap) // div) * div
        else:
            return len(self.labels)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.pocket_dataset.set_epoch(epoch)
        self.mol_dataset.set_epoch(epoch)
        super().set_epoch(epoch)

    def collater(self, samples):
        ret_pocket = []
        ret_lig = []
        batch_list = []
        act_list = []
        uniprot_list = []
        ret_protein = []
        assay_id_list = []

        if len(samples) == 0:
            return {}
        for pocket, ligs, acts, uniprot, assay_id, prot_seq in samples:
            ret_pocket.append(pocket)
            lignum_old = len(ret_lig)
            ret_lig += ligs
            batch_list.append([lignum_old, len(ret_lig)])
            uniprot_list.append(self.uniprot_id_dict[uniprot])
            assay_id_list.append(assay_id)
            act_list.append(acts)
            ret_protein.append(prot_seq)

        ret_pocket = self.pocket_dataset.collater(ret_pocket)
        ret_lig = self.mol_dataset.collater(ret_lig)
        return {"pocket": ret_pocket, "lig": ret_lig, "protein": ret_protein,
                "batch_list": batch_list, "act_list": act_list,
                "uniprot_list": uniprot_list, "assay_id_list": assay_id_list}

    # @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if self.split == "train":
            t_idx = self.trainidxmap[idx]
        else:
            t_idx = idx

        with data_utils.numpy_seed(1111, idx, self.epoch):
            pocket_name = np.random.choice(self.labels[t_idx]["pockets"], 1, replace=False)[0]

        lig_info = self.labels[t_idx]["ligands"]
        lig_info = [x for x in lig_info if x["smi"] in self.mol_smi2idx]
        uniprot = self.labels[t_idx]["uniprot"]
        assay_id = self.labels[t_idx].get("assay_id", "none")
        prot_seq = self.labels[t_idx]["sequence"]
        if len(lig_info) > self.max_lignum:
            with data_utils.numpy_seed(1111, idx, self.epoch):
                lig_idxes = np.random.choice(list(range(len(lig_info))), self.max_lignum, replace=False)
                lig_idxes = sorted(lig_idxes)
                lig_info = [lig_info[idx] for idx in lig_idxes]

        lig_idxes = [self.mol_smi2idx[info["smi"]] for info in lig_info]
        pocket_idx = self.pocket_name2idx[pocket_name]
        lig_act = [info["act"] for info in lig_info]
        pocket_data = self.pocket_dataset[pocket_idx]
        lig_data = [self.mol_dataset[x] for x in lig_idxes]

        return pocket_data, lig_data, lig_act, uniprot, assay_id, prot_seq