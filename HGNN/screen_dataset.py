import json
import os
import math
import numpy as np
import contextlib
import copy
import torch
from torch.utils.data import Dataset, sampler, DataLoader


def load_ligname():
    pdbbind_lig_dict = {}
    with open("./data/PDBbind_v2020/index/INDEX_general_PL_data.2020") as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            line = line.strip().split()
            lig = line[-1][1:-1]
            if lig != "":
                pdbid = line[0]
                pdbbind_lig_dict[pdbid] = lig
            else:
                continue

    with open("./data/PDBbind_v2020/index/INDEX_refined_data.2020") as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            line = line.strip().split()
            lig = line[-1][1:-1]
            if lig != "":
                pdbid = line[0]
                pdbbind_lig_dict[pdbid] = lig
            else:
                continue
    return pdbbind_lig_dict

def load_uniprotid():
    uniprot_id_dict = {}
    with open("./data/PDBbind_v2020/index/INDEX_refined_name.2020") as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            line = line.strip().split()
            uniprot_id = line[2]
            if uniprot_id != "" and uniprot_id != "------":
                pdbid = line[0]
                uniprot_id_dict[pdbid] = uniprot_id

    with open("./data/PDBbind_v2020/index/INDEX_general_PL_name.2020") as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            line = line.strip().split()
            uniprot_id = line[2]
            if uniprot_id != "" and uniprot_id != "------":
                pdbid = line[0]
                uniprot_id_dict[pdbid] = uniprot_id

    return uniprot_id_dict

def load_pocket_dude(result_root):
    data_root = f"{result_root}/DUDE"
    dude_pocket_feat = []
    dude_pocket_name = []
    for target in sorted(list(os.listdir(data_root))):
        pocket_arr = np.load(f"{data_root}/{target}/saved_target_embed.npy", allow_pickle=True)
        dude_pocket_feat.append(pocket_arr)
        dude_pocket_name.append(target)

    dude_pocket_feat = np.concatenate(dude_pocket_feat, axis=0)
    return dude_pocket_feat, dude_pocket_name

def load_pocket_dekois(result_root):
    data_root = f"{result_root}/DEKOIS"
    dekois_pocket_feat = []
    dekois_pocket_name = []
    for target in sorted(list(os.listdir(data_root))):
        pocket_arr = np.load(f"{data_root}/{target}/saved_target_embed.npy", allow_pickle=True)
        dekois_pocket_feat.append(pocket_arr)
        dekois_pocket_name.append(target)

    dekois_pocket_feat = np.concatenate(dekois_pocket_feat, axis=0)
    return dekois_pocket_feat, dekois_pocket_name

def load_pocket_pcba(result_root):
    data_root = f"{result_root}/PCBA"
    pcba_pocket_feat = []
    pcba_pocket_name = []
    for target in sorted(list(os.listdir(data_root))):
        pocket_arr = np.load(f"{data_root}/{target}/saved_target_embed.npy", allow_pickle=True)
        names_target = []
        for names in json.load(open(f"{data_root}/{target}/saved_pocket_names.json")):
            names_target += [ f"{target}_{x}" for x in names]

        if pocket_arr.shape[0] == 1:
            pocket_arr = np.concatenate([pocket_arr]*len(names_target), axis=0)
        pcba_pocket_feat.append(pocket_arr)
        pcba_pocket_name += names_target

    pcba_pocket_feat = np.concatenate(pcba_pocket_feat, axis=0)
    return pcba_pocket_feat, pcba_pocket_name


def read_cluster_file(cluster_file):
    protein_clstr_dict = {}
    with open(cluster_file) as f:
        line_in_clstr = []
        for line in f.readlines():
            if line.startswith(">"):
                for a in line_in_clstr:
                    for b in line_in_clstr:
                        if a not in protein_clstr_dict.keys():
                            protein_clstr_dict[a] = []
                        protein_clstr_dict[a].append(b)

                line_in_clstr = []
            else:
                line_in_clstr.append(line.split('|')[1])
    return protein_clstr_dict

def load_assayinfo(data_root, result_root):
    labels = json.load(open(f"{data_root}/train_label_pdbbind_seq.json")) + \
             json.load(open("../test_datasets/casf_label_seq.json"))
    save_dir_bdb = f"{result_root}/BDB"
    bdb_mol_smi = json.load(open(f"{save_dir_bdb}/bdb_mol_smis.json"))
    bdb_mol_smi = set(bdb_mol_smi)
    for label in labels:
        label["assay_id"] = label["pockets"][0].split("_")[0]
        label["domain"] = "pdbbind"

    # breakpoint()
    labels_bdb = json.load(open(f"{data_root}/train_label_blend_seq_full.json"))
    non_repeat_uniprot = []
    testset_uniport_root = "../test_datasets"
    non_repeat_uniprot += [x[0] for x in json.load(open(f"{testset_uniport_root}/dude.json"))]
    non_repeat_uniprot += [x[0] for x in json.load(open(f"{testset_uniport_root}/PCBA.json"))]
    non_repeat_uniprot += [x[0] for x in json.load(open(f"{testset_uniport_root}/dekois.json"))]
    non_repeat_uniprot_strict = []
    protein_clstr_dict_40 = read_cluster_file(f"{data_root}/uniport40.clstr")
    protein_clstr_dict_80 = read_cluster_file(f"{data_root}/uniport80.clstr")
    for uniprot in non_repeat_uniprot:
        non_repeat_uniprot_strict += protein_clstr_dict_80.get(uniprot, [])
        non_repeat_uniprot_strict.append(uniprot)
    old_len = len(labels_bdb)
    non_repeat_assayids = json.load(open(os.path.join(data_root, "fep_assays.json")))
    labels_bdb = [x for x in labels_bdb if (x["assay_id"] not in non_repeat_assayids)]
    labels_bdb = [x for x in labels_bdb if (x["uniprot"] not in non_repeat_uniprot)]

    labels_bdb_new = []
    for label in labels_bdb:
        ligands = label["ligands"]
        ligands_new = []
        for lig in ligands:
            if lig["smi"] in bdb_mol_smi and lig["act"] >= 5:
                ligands_new.append(lig)
        label["ligands"] = ligands_new
        if len(ligands_new) > 0:
            labels_bdb_new.append(label)

    labels += labels_bdb_new
    for label in labels:
        label["ligands"] = sorted(label["ligands"], key=lambda x: x["act"], reverse=True)

    # labels = [x for x in labels if (x["uniprot"] not in non_repeat_uniprot_strict)]
    return labels

def load_id_dict(result_root, assayinfo_lst):
    import random
    random.seed(42)
    bdb_dir = f"{result_root}/BDB"
    pdbbind_dir = f"{result_root}/PDBBind"

    pocket_names = json.load(open(f"{bdb_dir}/bdb_pocket_names.json"))
    pocket_embed = np.load(f"{bdb_dir}/bdb_pocket_reps.npy")
    name2idx = {name:i for i, name in enumerate(pocket_names)}

    assay_feat_lst = []
    bdb_assayid_lst = []
    for assay in assayinfo_lst:
        assay_id = assay["assay_id"]
        if assay.get("domain", None) == "pdbbind":
            continue
        pockets = assay["pockets"]
        repeat_num = len(assay["ligands"])
        repeat_num = int(np.sqrt(repeat_num))
        for i in range(repeat_num):
            pocket = random.choice(pockets)
            assay_feat_lst.append(pocket_embed[name2idx[pocket]])
            bdb_assayid_lst.append(assay_id)

    bdb_assay_feat = np.stack(assay_feat_lst)

    train_pdbbind_ids = json.load(open(f'{pdbbind_dir}/train_pdbbind_ids.json'))
    train_pdbbind_pocket_embed = np.load(f"{pdbbind_dir}/train_pocket_reps.npy")
    train_pdbbind_ids_new = []
    train_pdbbind_pocket_embed_new = []
    pdbbind_aidlist = [assay["assay_id"] for assay in assayinfo_lst if assay.get("domain", None) == "pdbbind"]
    pdbbind_aidset = set(pdbbind_aidlist)
    for id, embed in zip(train_pdbbind_ids, train_pdbbind_pocket_embed):
        if id in pdbbind_aidset:
            train_pdbbind_ids_new.append(id)
            train_pdbbind_pocket_embed_new.append(embed)

    train_pdbbind_ids = train_pdbbind_ids_new
    train_pdbbind_pocket_embed = np.stack(train_pdbbind_pocket_embed_new)

    train_pocket = bdb_assayid_lst + train_pdbbind_ids
    pocket_feat_train = np.concatenate([bdb_assay_feat, train_pdbbind_pocket_embed])
    test_pocket = json.load(open(f'{pdbbind_dir}/test_pdbbind_ids.json'))
    pocket_feat_test = np.load(f'{pdbbind_dir}/test_pocket_reps.npy')

    return train_pocket, test_pocket, pocket_feat_train, pocket_feat_test


def load_datas(data_root, result_root):
    assayinfo_lst = load_assayinfo(data_root, result_root)
    assayid_lst_train, assayid_lst_test, pocket_feat_train, pocket_feat_test = load_id_dict(result_root, assayinfo_lst)

    dude_pocket_feat, dude_pocket_name = load_pocket_dude(result_root)

    pcba_pocket_feat, pcba_pocket_name = load_pocket_pcba(result_root)

    dekois_pocket_feat, dekois_pocket_name = load_pocket_dekois(result_root)

    pocket_feat = np.concatenate((pocket_feat_train, pocket_feat_test, dude_pocket_feat, pcba_pocket_feat, dekois_pocket_feat), axis=0)
    assayid_lst_all = assayid_lst_train + assayid_lst_test + dude_pocket_name + pcba_pocket_name + dekois_pocket_name

    save_dir_bdb = f"{result_root}/BDB"
    save_dir_pdbbind = f"{result_root}/PDBBind"
    mol_feat_train_bdb = np.load(f'{save_dir_bdb}/bdb_mol_reps.npy')
    mol_feat_train_pdbbind = np.load(f'{save_dir_pdbbind}/train_mol_reps.npy')
    mol_feat_test = np.load(f'{save_dir_pdbbind}/test_mol_reps.npy')
    mol_feat = np.concatenate((mol_feat_train_bdb, mol_feat_train_pdbbind, mol_feat_test), axis=0)
    mol_smi_lst = json.load(open(f"{save_dir_bdb}/bdb_mol_smis.json")) + json.load(open(f"{save_dir_pdbbind}/train_mol_smis.json")) + json.load(open(f"{save_dir_pdbbind}/test_mol_smis.json"))
    test_len = len(json.load(open(f"{save_dir_pdbbind}/test_mol_smis.json")))
    test_molidxes = range(len(mol_smi_lst)-test_len, len(mol_smi_lst))
    return assayinfo_lst, pocket_feat, mol_feat, assayid_lst_all, mol_smi_lst, \
           assayid_lst_train, assayid_lst_test, dude_pocket_name, pcba_pocket_name, dekois_pocket_name, test_molidxes

def load_valid_label(assayid_lst_test):
    coreset = list(open("./data/CoreSet.dat").readlines())[1:]
    pdbid2cluster = {}
    for line in coreset:
        line = line.strip().split()
        pdbid = line[0]
        cluster = line[-1]
        pdbid2cluster[pdbid] = cluster

    labels = np.zeros((len(assayid_lst_test), len(assayid_lst_test)))
    for i, pdbid_1 in enumerate(assayid_lst_test):
        for j, pdbid_2 in enumerate(assayid_lst_test):
            if pdbid2cluster[pdbid_1] != pdbid2cluster[pdbid_2]:
                labels[i, j] = 0
            else:
                labels[i, j] = 1
    return labels

def load_pocket_pocket_graph(data_root, assayid_lst_all, assayid_lst_train):
    neighbor_dict_train = json.load(
        open(f"{data_root}/align_pair_res_train_10.23.json"))
    train_keys = json.load(
        open(f"{data_root}/align_train_keys_10.23.json"))
    neighbor_dict_train_new = {}
    for idx, neighbors in neighbor_dict_train.items():
        neighbor_dict_train_new[train_keys[int(idx)]] = neighbors
    neighbor_dict_train = neighbor_dict_train_new
    assayid_set = set(assayid_lst_all)
    assayid_set_train = set(assayid_lst_train)
    PPGraph = {}

    for assayid_1 in neighbor_dict_train.keys():
        if assayid_1 not in assayid_set:
            continue
        neighbor_dict_train[assayid_1] = sorted(neighbor_dict_train[assayid_1], key=lambda x: x[1], reverse=True)

        score_new = []
        for assayid_2, score in neighbor_dict_train[assayid_1]:
            if assayid_2 not in assayid_set_train:
                continue
            if score < 0.5:
                continue
            score_new.append((assayid_2, score))
        PPGraph[assayid_1] = score_new

    import pickle
    align_res_test = json.load(open(f"{data_root}/align_pair_res_test_10.23.json"))
    align_score_test = {}

    for test_id in align_res_test.keys():
        if test_id not in assayid_set:
            continue
        pocket_sim_infos = align_res_test[test_id]
        pocket_sim_infos = sorted(pocket_sim_infos, key=lambda x: x[1], reverse=True)
        score_new = []
        for test_target, score in pocket_sim_infos:
            test_target = test_target.split('.')[0]
            if test_target not in assayid_set_train:
                continue
            if score < 0.5:
                continue
            score_new.append((test_target, score))
        align_score_test[test_id] = score_new

    # breakpoint()
    PPGraph = {**PPGraph, **align_score_test}
    return PPGraph

@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

class ScreenDataset(Dataset):
    def __init__(self, batch_size, assay_graph, assayinfo_lst, assayid_lst_all, mol_smi_lst, assayid_lst_train):
        self.batch_size = batch_size
        self.train_idxes = list(range(len(assayid_lst_train)))
        self.assayid_set_train = set(assayid_lst_train)
        self.train_idxes_epoch = copy.deepcopy(self.train_idxes)
        self.assay_graph = assay_graph
        self.assayinfo_dicts = {x["assay_id"]: x for x in assayinfo_lst}
        self.smi2idx = {smi:idx for idx, smi in enumerate(mol_smi_lst)}
        self.uniprotid_dict = load_uniprotid()
        self.pocket_lig_graph = self.load_graph()
        self.seed = 66
        self.assayid2idxes = {}
        for idx, assayid in enumerate(assayid_lst_all):
            if assayid not in self.assayid2idxes:
                self.assayid2idxes[assayid] = []
            self.assayid2idxes[assayid].append(idx)
        self.idx2assayid = assayid_lst_all
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        with numpy_seed(self.seed, epoch):
            self.train_idxes_epoch = copy.deepcopy(self.train_idxes)
            np.random.shuffle(self.train_idxes_epoch)

    def load_graph(self):
        pocket_lig_graph = {}
        if os.path.exists("./data/pocket_lig_graph.json"):
            pocket_lig_graph = json.load(open("./data/pocket_lig_graph.json"))
        else:
            from tqdm import tqdm
            for assayid in tqdm(self.assayid2idxes.keys()):
                if assayid not in self.assayid_set_train:
                    continue
                ligands = self.assayinfo_dicts[assayid]["ligands"]
                lig_candidate = []
                if len(ligands) > 1:
                    lig_assay = [x["smi"] for x in ligands if x["act"] >= 5]
                else:
                    lig_assay = [x["smi"] for x in ligands]
                lig_candidate += lig_assay
                lig_assay = set(lig_assay)
                uniprot = self.assayinfo_dicts[assayid]["uniprot"]

                for assayid_nbr, score in self.assay_graph.get(assayid, []):
                    if assayid_nbr not in self.assayinfo_dicts:
                        continue
                    assay_nbr = self.assayinfo_dicts[assayid_nbr]
                    uniprot_nbr = assay_nbr["uniprot"]
                    ligands_nbr = assay_nbr["ligands"]
                    if len(ligands) > 1:
                        lig_candidate_nbr = [x["smi"] for x in ligands_nbr if x["act"] >= 5]
                    else:
                        lig_candidate_nbr = [x["smi"] for x in ligands_nbr]
                    if assayid_nbr not in self.assayid_set_train:
                        continue
                    if len(lig_assay & set(lig_candidate_nbr)) > 0:
                        lig_candidate += lig_candidate_nbr
                    elif uniprot == uniprot_nbr:
                        lig_candidate += lig_candidate_nbr

                pocket_lig_graph[assayid] = [x for x in set(lig_candidate) if x in self.smi2idx]

            json.dump(pocket_lig_graph, open("./data/pocket_lig_graph.json", "w"))
        return pocket_lig_graph

    def __getitem__(self, item):
        pocket_idx_batch = self.train_idxes_epoch[item*self.batch_size:(item+1)*self.batch_size]
        pocket_batch = [self.idx2assayid[idx] for idx in pocket_idx_batch]
        lig_batch = []
        lig_idx_batch = []
        epoch = self.epoch
        for pocket in pocket_batch:
            lig_candidate = self.pocket_lig_graph[pocket]
            with numpy_seed(self.seed, epoch, item):
                lig = np.random.choice(lig_candidate)
                lig_batch.append(lig)

            lig_idx_batch.append(self.smi2idx[lig])

        labels = np.zeros((self.batch_size, self.batch_size))
        for i, pocket in enumerate(pocket_batch):
            for j, lig in enumerate(lig_batch):
                if lig in self.pocket_lig_graph[pocket]:
                    labels[i, j] = 1
                else:
                    labels[i, j] = 0

        return torch.tensor(pocket_idx_batch), torch.tensor(lig_idx_batch), torch.tensor(labels)

    def __len__(self):
        return len(self.train_idxes_epoch) // self.batch_size



