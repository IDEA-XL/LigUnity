# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json

# from IPython import embed as debug_embedded
import logging
import os
import random
from datetime import datetime
# from collections.abc import Iterable
# from sklearn.metrics import roc_auc_score
from xmlrpc.client import Boolean
import numpy as np
import torch
import pickle
from typing import Dict, List, Set, Tuple, Union
from tqdm import tqdm
from unicore import checkpoint_utils
import unicore
from unicore.data import (AppendTokenDataset, Dictionary, EpochShuffleDataset,
                          FromNumpyDataset, NestedDictionaryDataset,
                          PrependTokenDataset, RawArrayDataset,LMDBDataset, RawLabelDataset,
                          RightPadDataset, RightPadDataset2D, TokenizeDataset,SortDataset,data_utils)
from unicore.tasks import UnicoreTask, register_task
from unimol.data import (PairDataset, AffinityDataset, CroppingPocketDataset,CroppingDataset,
                         CrossDistanceDataset, DistanceDataset,
                         EdgeTypeDataset, KeyDataset, LengthDataset,
                         NormalizeDataset, NormalizeDockingPoseDataset,
                         PrependAndAppend2DDataset, RemoveHydrogenDataset,
                         RemoveHydrogenPocketDataset, RightPadDatasetCoord,
                         RightPadDatasetCross2D, TTADockingPoseDataset, AffinityTestDataset, AffinityValidDataset, AffinityMolDataset, AffinityPocketDataset, ResamplingDataset)
#from skchem.metrics import bedroc_score
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment
from sklearn.metrics import roc_curve
logger = logging.getLogger(__name__)
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_mol(s: str, keep_h: bool, add_h: bool, keep_atom_map: bool):
    """
    Builds an RDKit molecule from a SMILES string.

    :param s: SMILES string.
    :param keep_h: Boolean whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
    :param add_h: Boolean whether to add hydrogens to the input smiles.
    :param keep_atom_map: Boolean whether to keep the original atom mapping.
    :return: RDKit molecule.
    """
    params = Chem.SmilesParserParams()
    params.removeHs = not keep_h if not keep_atom_map else False
    mol = Chem.MolFromSmiles(s, params)

    if add_h:
        mol = Chem.AddHs(mol)

    if keep_atom_map:
        atom_map_numbers = tuple(atom.GetAtomMapNum() for atom in mol.GetAtoms())
        for idx, map_num in enumerate(atom_map_numbers):
            if idx + 1 != map_num:
                new_order = np.argsort(atom_map_numbers).tolist()
                return Chem.rdmolops.RenumberAtoms(mol, new_order)

    return mol

def generate_scaffold(mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]], include_chirality: bool = False) -> str:
    """
    Computes the Bemis-Murcko scaffold for a SMILES string.
    :param mol: A SMILES or an RDKit molecule.
    :param include_chirality: Whether to include chirality in the computed scaffold..
    :return: The Bemis-Murcko scaffold for the molecule.
    """
    if isinstance(mol, str):
        mol = make_mol(mol, keep_h=False, add_h=False, keep_atom_map=False)
    if isinstance(mol, tuple):
        mol = mol[0]
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold


def scaffold_split(smi_list, num_sup, seed=1):
    scaffold_dict = {}
    for i, smi in enumerate(smi_list):
        scaffold = generate_scaffold(smi)
        if scaffold not in scaffold_dict:
            scaffold_dict[scaffold] = []
        scaffold_dict[scaffold].append(i)
    scaffold_id_list = [(k, v) for k, v in scaffold_dict.items()]
    random.seed(seed)
    random.shuffle(scaffold_id_list)
    # print([len(x[1]) for x in scaffold_id_list])
    # scaffold_id_list = sorted(scaffold_id_list, key=lambda x: len(x[1]))
    idx_list_all = []
    for scaffold, idx_list in scaffold_id_list:
        idx_list_all += idx_list

    return idx_list_all
    

def is_older(_version):
    if isinstance(_version, int):
        return _version <= 25
    else:
        dt1 = datetime.strptime("2019-03-01 00:00:00", "%Y-%m-%d %H:%M:%S")
        dt2 = datetime.strptime(_version, "%Y-%m-%d %H:%M:%S")
        return dt2 < dt1

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

@register_task("train_task")
class pocketscreen(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="downstream data path",
        )
        parser.add_argument(
            "--finetune-mol-model",
            default=None,
            type=str,
            help="pretrained molecular model path",
        )
        parser.add_argument(
            "--finetune-pocket-model",
            default=None,
            type=str,
            help="pretrained pocket model path",
        )
        parser.add_argument(
            "--restore-model",
            default=None,
            type=str,
            help="pretrained pocket model path",
        )
        parser.add_argument(
            "--dist-threshold",
            type=float,
            default=6.0,
            help="threshold for the distance between the molecule and the pocket",
        )
        parser.add_argument(
            "--max-pocket-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a pocket",
        )
        parser.add_argument(
            "--test-model",
            default=False,
            type=Boolean,
            help="whether test model",
        )
        parser.add_argument("--reg", action="store_true", help="regression task")
        parser.add_argument(
            "--few-shot",
            default=False,
            type=Boolean,
            help="whether few-shot testing",
        )
        parser.add_argument(
            "--sup-num",
            default=16,
            type=float
        )
        parser.add_argument(
            "--valid-set",
            default="CASF",
            type=str
        )
        parser.add_argument(
            "--max-lignum",
            type=int,
            default=16
        )
        parser.add_argument(
            "--test-max-lignum",
            type=int,
            default=512
        )
        parser.add_argument(
            "--split-method",
            type=str,
            default="random",
            help="split method for few-shot testing",
        )
        parser.add_argument(
            "--results-path",
            type=str,
            default=None,
            help="results path for few-shot testing",
        )
        parser.add_argument(
            "--assay-idx",
            type=int,
            default=0
        )
        parser.add_argument(
            "--contras-weight",
            type=float,
            default=0.5
        )
        parser.add_argument(
            "--rank-weight",
            type=float,
            default=0.5
        )
        parser.add_argument(
            "--protein-similarity-thres",
            type=float,
            default=1.0
        )
        parser.add_argument(
            "--neg-margin",
            type=float,
            default=2.0
        )
        parser.add_argument(
            "--demo-lig-file",
            type=str,
            default=""
        )
        parser.add_argument(
            "--demo-prot-file",
            type=str,
            default=""
        )
        parser.add_argument(
            "--demo-split-file",
            type=str,
            default=""
        )
        parser.add_argument(
            "--case-train-ligfile",
            type=str,
            default=""
        )
        parser.add_argument(
            "--case-test-ligfile",
            type=str,
            default=""
        )
        parser.add_argument(
            "--active-learning-resfile",
            type=str,
            default=""
        )

    def __init__(self, args, dictionary, pocket_dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.pocket_dictionary = pocket_dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        self.pocket_mask_idx = pocket_dictionary.add_symbol("[MASK]", is_special=True)
        self.mol_reps = None
        self.keys = None

    def reduce_metrics(self, logging_outputs, loss, split='train'):
        """Aggregate logging outputs from data parallel training."""
        from unicore import metrics, utils
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        metrics.log_scalar("bsz", bsz, priority=190, round=1)

        loss.__class__.reduce_metrics(logging_outputs, split, self.args)

    @classmethod
    def setup_task(cls, args, **kwargs):
        mol_dictionary = Dictionary.load(os.path.join(PROJECT_ROOT, "vocab", "dict_mol.txt"))
        pocket_dictionary = Dictionary.load(os.path.join(PROJECT_ROOT, "vocab", "dict_pkt.txt"))
        logger.info("ligand dictionary: {} types".format(len(mol_dictionary)))
        logger.info("pocket dictionary: {} types".format(len(pocket_dictionary)))
        return cls(args, mol_dictionary, pocket_dictionary)

    def load_few_shot_demo_dataset(self, split, **kwargs):
        ligands_lmdb = os.path.join(self.args.demo_lig_file)
        pocket_lmdb = os.path.join(self.args.demo_prot_file)
        split_info = json.load(open(self.args.demo_split_file))
        import copy
        pair_label = copy.deepcopy(split_info)

        if split == "train":
            pair_label["ligands"] = [lig for lig in split_info["train"]]
            print("number of training ligands", len(pair_label["ligands"]))
        else:
            pair_label["ligands"] = [lig for lig in split_info["test"]]
            print("number of testing ligands", len(pair_label["ligands"]))
        pair_label["ligands"] = sorted(pair_label["ligands"], key=lambda x: x["act"], reverse=True)

        pocket_dataset = self.load_pockets_dataset(pocket_lmdb, is_train=split=="train")
        mol_dataset = self.load_mols_dataset(ligands_lmdb, "atoms", "coordinates", is_train=split=="train")
        dataset = PairDataset(self.args, pocket_dataset, mol_dataset, [pair_label], split, use_cache=False)

        self.datasets[split] = dataset
        return dataset

    def load_few_shot_TYK2_FEP_dataset(self, split, **kwargs):
        test_datasets_root = os.path.join(PROJECT_ROOT, "test_datasets")
        save_path = f"{test_datasets_root}/FEP"
        ligands_lmdb = os.path.join(f"{test_datasets_root}/case_study/tyk2_fep_ligands.lmdb")
        pocket_lmdb = os.path.join(f"{test_datasets_root}/FEP/proteins.lmdb")
        pair_label_all = json.load(open(f"{test_datasets_root}/case_study/tyk2_fep.json"))
        pair_label_all = [pair_label_all]
        import pandas as pd

        train_smiles = set(pd.read_csv(self.args.case_train_ligfile)['Smiles'].tolist())
        test_smiles = set(pd.read_csv(self.args.case_test_ligfile)['Smiles'].tolist())

        act_all = []
        avgact_train = 6.955628350893639
        for pair_label in pair_label_all:
            pair_label["assay_id"] = "tyk2"
            act_all += [x["act"] for x in pair_label["ligands"]]
            if split == "train":
                pair_label["ligands"] = [lig for lig in pair_label["ligands"] if lig["smi"] in train_smiles]
                print("number of training ligands", len(pair_label["ligands"]))
            else:
                pair_label["ligands"] = [lig for lig in pair_label["ligands"] if lig["smi"] in test_smiles]
                print("number of testing ligands", len(pair_label["ligands"]))
            pair_label["ligands"] = sorted(pair_label["ligands"], key=lambda x: x["act"], reverse=True)

        print("average activity of tyk2:", np.mean(act_all))
        print("moving the average to be", avgact_train)
        for assay in pair_label_all:
            for lig in assay["ligands"]:
                lig["act"] = (lig["act"] - np.mean(act_all))/np.std(act_all) + avgact_train

        pocket_dataset = self.load_pockets_dataset(pocket_lmdb, is_train=split=="train")
        mol_dataset = self.load_mols_dataset(ligands_lmdb, "atoms", "coordinates", is_train=split=="train")
        dataset = PairDataset(self.args, pocket_dataset, mol_dataset, pair_label_all, split, use_cache=False)

        self.datasets[split] = dataset
        return dataset

    def load_few_shot_FEP_dataset(self, split, **kwargs):
        data_path = os.path.join(PROJECT_ROOT, "test_datasets", "FEP")
        ligands_lmdb = os.path.join(f"{data_path}/ligands.lmdb")
        pocket_lmdb = os.path.join(f"{data_path}/proteins.lmdb")
        pair_label_all = json.load(open(f"{data_path}/fep_labels.json"))

        for pair_label in pair_label_all:
            pair_label["assay_id"] = pair_label["uniprot"]
            if self.args.sup_num < 1:
                k_shot = int(self.args.sup_num * len(pair_label["ligands"]))
            else:
                k_shot = int(self.args.sup_num)
            random.seed(self.args.seed)
            random.shuffle(pair_label["ligands"])
            if split == "train":
                pair_label["ligands"] = sorted(pair_label["ligands"][:k_shot], key=lambda x: x["act"], reverse=True)
            else:
                pair_label["ligands"] = sorted(pair_label["ligands"][k_shot:], key=lambda x: x["act"], reverse=True)

        pocket_dataset = self.load_pockets_dataset(pocket_lmdb, is_train=split=="train")
        mol_dataset = self.load_mols_dataset(ligands_lmdb, "atoms", "coordinates", is_train=split=="train")
        dataset = PairDataset(self.args, pocket_dataset, mol_dataset, pair_label_all, split, use_cache=False)

        self.datasets[split] = dataset
        return dataset

    def load_few_shot_ood_dataset(self, split, **kwargs):
        data_path = os.path.join(PROJECT_ROOT, "test_datasets", "OOD")
        mol_data_path = os.path.join(data_path, "oodtest_unit=%_lig.lmdb")
        pocket_data_path = os.path.join(data_path, "oodtest_unit=%_prot.lmdb")
        assay_test_ood = json.load(open(os.path.join(data_path, "oodtest_unit=%.json")))
        act_all = []
        avgact_train = 6.955628350893639
        for assay in assay_test_ood:
            if self.args.sup_num < 1:
                k_shot = int(self.args.sup_num * len(assay["ligands"]))
            else:
                k_shot = int(self.args.sup_num)

            if self.args.split_method == "random":
                select_index = list(range(len(assay["ligands"])))
                random.seed(self.args.seed)
                random.shuffle(select_index)
            elif self.args.split_method == "scaffold":
                smi_list = [x["smi"] for x in assay["ligands"]]
                select_index = scaffold_split(smi_list, k_shot)
            else:
                raise ValueError(f"Invalid split method: {self.args.split_method}. Supported methods are 'random' and 'scaffold'")

            if split == "train":
                assay["ligands"] = [assay["ligands"][idx] for idx in select_index[:k_shot]]
            else:
                assay["ligands"] = [assay["ligands"][idx] for idx in select_index[k_shot:]]
            assay["ligands"] = [{"smi":x["smi"], "act":-x["act"]} for x in assay["ligands"]]
            assay["ligands"] = sorted(assay["ligands"], key=lambda x: x["act"], reverse=True)
            act_all += [x["act"] for x in assay["ligands"]]

        print("average activity of ood:", np.mean(act_all))
        print("moving the average to be", avgact_train)
        for assay in assay_test_ood:
            for lig in assay["ligands"]:
                lig["act"] = lig["act"] - np.mean(act_all) + avgact_train

        pocket_dataset = self.load_pockets_dataset(pocket_data_path, is_train=split=="train")
        mol_dataset = self.load_mols_dataset(mol_data_path, "atoms", "coordinates", is_train=split=="train")
        dataset = PairDataset(self.args, pocket_dataset, mol_dataset, assay_test_ood, split, use_cache=False)

        self.datasets[split] = dataset
        return dataset


    def load_few_shot_timesplit(self, split, **kwargs):
        test_datasets_root = os.path.join(PROJECT_ROOT, "test_datasets")
        mol_data_path = os.path.join(test_datasets_root, "TIME", "test_lig_timesplit.lmdb")
        pocket_data_path = os.path.join(test_datasets_root, "TIME", "test_prot_timesplit.lmdb")
        test_assays = json.load(open(os.path.join(test_datasets_root, "TIME", "assay_test_timesplit.json")))
        
        print("number of test assays", len(test_assays))
        for assay in test_assays:
            if self.args.sup_num < 1:
                k_shot = int(self.args.sup_num * len(assay["ligands"]))
            else:
                k_shot = int(self.args.sup_num)

            if self.args.split_method == "random":
                select_index = list(range(len(assay["ligands"])))
                random.seed(self.args.seed)
                random.shuffle(select_index)
            elif self.args.split_method == "scaffold":
                smi_list = [x["smi"] for x in assay["ligands"]]
                select_index = scaffold_split(smi_list, k_shot, self.args.seed)
            else:
                raise ValueError(f"Invalid split method: {self.args.split_method}. Supported methods are 'random' and 'scaffold'")

            if split == "train":
                assay["ligands"] = [assay["ligands"][idx] for idx in select_index[:k_shot]]
            else:
                assay["ligands"] = [assay["ligands"][idx] for idx in select_index[k_shot:]]
            assay["ligands"] = sorted(assay["ligands"], key=lambda x: x["act"], reverse=True)

        pocket_dataset = self.load_pockets_dataset(pocket_data_path, is_train=split=="train")
        mol_dataset = self.load_mols_dataset(mol_data_path, "atoms", "coordinates", is_train=split=="train")
        dataset = PairDataset(self.args, pocket_dataset, mol_dataset, assay_test_unseen, split, use_cache=True)

        self.datasets[split] = dataset
        return dataset


    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        'smi','pocket','atoms','coordinates','pocket_atoms','pocket_coordinates'
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """

        if self.args.few_shot:
            if self.args.valid_set == "TYK2":
                return self.load_few_shot_TYK2_FEP_dataset(split, **kwargs)
            elif self.args.valid_set == "FEP":
                return self.load_few_shot_FEP_dataset(split, **kwargs)
            elif self.args.valid_set == "TIME":
                return self.load_few_shot_timesplit(split, **kwargs)
            elif self.args.valid_set == "OOD":
                return self.load_few_shot_ood_dataset(split, **kwargs)
            elif self.args.valid_set == "DEMO":
                return self.load_few_shot_demo_dataset(split, **kwargs)

        protein_clstr_dict = {}
        if self.args.protein_similarity_thres == 0.4:
            protein_clstr_dict_40 = read_cluster_file(
                f"{self.args.data}/uniport40.clstr")
            protein_clstr_dict = protein_clstr_dict_40
        elif self.args.protein_similarity_thres == 0.8:
            protein_clstr_dict_80 = read_cluster_file(
                f"{self.args.data}/uniport80.clstr")
            protein_clstr_dict = protein_clstr_dict_80

        if split == "train" or (split == "valid" and self.args.valid_set == "TIME"):
            mol_data_path = os.path.join(self.args.data, "train_lig_all_blend.lmdb")
            pocket_data_path = os.path.join(self.args.data, "train_prot_all_blend.lmdb")
            mol_dataset = self.load_mols_dataset(mol_data_path, "atoms", "coordinates", is_train=split=="train")
            pocket_dataset = self.load_pockets_dataset(pocket_data_path, is_train=split=="train")
            pair_label_1 = json.load(open(os.path.join(self.args.data, "train_label_pdbbind_seq.json")))
            pair_label_2 = json.load(open(os.path.join(self.args.data, "train_label_blend_seq_full.json")))
            test_datasets_root = os.path.join(PROJECT_ROOT, "test_datasets")
            if self.args.valid_set == "TIME":
                pair_label_2_new = []
                for assay in pair_label_2:
                    version = assay["version"]
                    version_older = is_older(version)
                    if version_older and split == "train":
                        pair_label_2_new.append(assay)
                    elif (not version_older) and split == "valid":
                        lig_new = [lig for lig in assay["ligands"] if lig["rel"] == "="]
                        assay["ligands"] = lig_new
                        if len(assay["ligands"]) >= 10:
                            pair_label_2_new.append(assay)
                pair_label_2 = pair_label_2_new
            else:
                repeat_ligands = json.load(open(os.path.join(test_datasets_root, "fep_repeat_ligands_can.json")))
                if "no_similar_ligand" in self.args.save_dir:
                    sim_ligands_cache = os.path.join(test_datasets_root, "fep_similar_ligands_0d5.json")
                    repeat_ligands += json.load(open(sim_ligands_cache))

                pair_label_2_new = []
                repeat_ligands = set(repeat_ligands)
                print("number of deleted ligands", len(repeat_ligands))
                for assay in pair_label_2:
                    ligands_new = []
                    for lig in assay["ligands"]:
                        if lig["smi"] not in repeat_ligands:
                            ligands_new.append(lig)
                    if len(ligands_new) >= 3:
                        assay["ligands"] = ligands_new
                        pair_label_2_new.append(assay)
                print("number of assays before processing :", len(pair_label_2))
                pair_label_2 = pair_label_2_new
                print("number of assays after remove ligands in FEP:", len(pair_label_2))

                non_repeat_assayids = json.load(open(os.path.join(test_datasets_root, "fep_assay_ids.json")))
                non_repeat_assayids = set(non_repeat_assayids)

                pair_label_2 = [x for x in pair_label_2 if (x["assay_id"] not in non_repeat_assayids)]
                print("number of assays after remove assays in FEP:", len(pair_label_2))

                if self.args.valid_set == "CASF":
                    # remove all testset protein by default
                    testset_uniprot_lst = []
                    testset_uniprot_lst += [x[0] for x in json.load(open(f"{test_datasets_root}/dude.json"))]
                    testset_uniprot_lst += [x[0] for x in json.load(open(f"{test_datasets_root}/PCBA.json"))]
                    testset_uniprot_lst += [x[0] for x in json.load(open(f"{test_datasets_root}/dekois.json"))]

                    # remove all similar protein
                    if "no_similar_protein" in self.args.save_dir:
                        testset_uniprot_lst_new = []
                        for uniprot in testset_uniprot_lst:
                            testset_uniprot_lst_new += protein_clstr_dict.get(uniprot, [uniprot])
                            testset_uniprot_lst_new.append(uniprot)
                        testset_uniprot_lst = testset_uniprot_lst_new
                    print(testset_uniprot_lst)
                elif self.args.valid_set == "FEP":
                    # don't remove similar protein by default (lead optimization sceneario)
                    testset_uniprot_lst = []

                    # remove all similar protein
                    if "no_similar_protein" in self.args.save_dir:
                        testset_uniprot_lst += [x[0] for x in json.load(open(f"{test_datasets_root}/FEP.json"))]
                        testset_uniprot_lst_new = []
                        for uniprot in testset_uniprot_lst:
                            testset_uniprot_lst_new += protein_clstr_dict.get(uniprot, [uniprot])
                            testset_uniprot_lst_new.append(uniprot)
                        testset_uniprot_lst = testset_uniprot_lst_new
                        print(testset_uniprot_lst)
                else:
                    testset_uniprot_lst = []

                pair_label_2 = [x for x in pair_label_2 if (x["uniprot"] not in testset_uniprot_lst)]
                print("number of assay after remove test uniport:", len(pair_label_2))

                # using dataset processed by DrugCLIP in 100% similarity threshold
                # remove all similar protein in PDBBind when testing on No similar protein setting
                if "no_similar_protein" in self.args.save_dir:
                    old_len = len(pair_label_1)
                    pair_label_1 = [x for x in pair_label_1 if (x["uniprot"] not in testset_uniprot_lst)]
                    print("number of deleted pdbbind after remove test uniport", old_len - len(pair_label_1))

            print(f"totally {len(pair_label_2)} datas (blend ChEMBL and BindingDB) for {split}")
            pair_label = pair_label_2
            if split == "train":
                pair_label += pair_label_1
            pair_dataset = PairDataset(self.args, pocket_dataset, mol_dataset, pair_label, split=split, use_cache=True, cache_dir=self.args.data)
        elif split == "valid" and self.args.valid_set == "CASF":
            # casf valid
            split_old = split
            split = "valid"
            mol_data_path = os.path.join(self.args.data, split + "_lig.lmdb")
            pocket_data_path = os.path.join(self.args.data, split + "_prot.lmdb")
            pair_label = json.load(open(os.path.join(self.args.data, split + "_label_seq.json")))
            split = split_old
            mol_dataset = self.load_mols_dataset(mol_data_path, "atoms", "coordinates")
            pocket_dataset = self.load_pockets_dataset(pocket_data_path)
            pair_dataset = PairDataset(self.args, pocket_dataset, mol_dataset, pair_label, split, use_cache=False)
        elif split == "valid" and self.args.valid_set == "FEP":
            # fep valid
            save_path = os.path.join(PROJECT_ROOT, "test_datasets", "FEP")
            mol_data_path = os.path.join(f"{save_path}/ligands.lmdb")
            pocket_data_path = os.path.join(f"{save_path}/proteins.lmdb")
            pair_label = json.load(open(f"{save_path}/fep_labels.json"))
            mol_dataset = self.load_mols_dataset(mol_data_path, "atoms", "coordinates")
            pocket_dataset = self.load_pockets_dataset(pocket_data_path)
            pair_dataset = PairDataset(self.args, pocket_dataset, mol_dataset, pair_label, split, use_cache=False)



        if split == "train":
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(pair_dataset))

            self.datasets[split] = SortDataset(
                pair_dataset,
                sort_order=[shuffle],
            )
            self.datasets[split] = ResamplingDataset(
                self.datasets[split]
            )
        else:
            self.datasets[split] = pair_dataset
        return pair_dataset

    def load_mols_dataset(self, data_path, atoms, coords, **kwargs):
        dataset = LMDBDataset(data_path)
        # label_dataset = KeyDataset(dataset, "label")
        dataset = AffinityMolDataset(
            dataset,
            self.args.seed,
            atoms,
            coords,
            is_train=kwargs.get("is_train", False),
        )

        smi_dataset = KeyDataset(dataset, "smi")
        if kwargs.get("load_name", False):
            name_dataset = KeyDataset(dataset, "name")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)
        dataset = CroppingDataset(
            dataset,
            self.seed,
            atoms,
            coords,
            self.args.max_pocket_atoms//4,
        )
        apo_dataset = NormalizeDataset(dataset, "coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        len_dataset = LengthDataset(src_dataset)
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(apo_dataset, "coordinates")
        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        distance_dataset = DistanceDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)

        in_datasets = {
            "net_input": {
                "mol_src_tokens": RightPadDataset(
                    src_dataset,
                    pad_idx=self.dictionary.pad(),
                ),
                "mol_src_distance": RightPadDataset2D(
                    distance_dataset,
                    pad_idx=0,
                ),
                "mol_src_edge_type": RightPadDataset2D(
                    edge_type,
                    pad_idx=0,
                ),
            },
            "smi_name": RawArrayDataset(smi_dataset),
            # "target":  RawArrayDataset(label_dataset),
            "mol_len": RawArrayDataset(len_dataset),
        }
        if kwargs.get("load_name", False):
            in_datasets["name"] = name_dataset

        nest_dataset = NestedDictionaryDataset(in_datasets)
        return nest_dataset

    def load_pockets_dataset(self, data_path, **kwargs):

        dataset = LMDBDataset(data_path)

        dataset = AffinityPocketDataset(
            dataset,
            self.args.seed,
            "pocket_atoms",
            "pocket_coordinates",
            is_train=kwargs.get("is_train", False),
            pocket="pocket"
        )
        poc_dataset = KeyDataset(dataset, "pocket")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenPocketDataset(
            dataset,
            "pocket_atoms",
            "pocket_coordinates",
            True,
            True,
        )
        dataset = CroppingPocketDataset(
            dataset,
            self.seed,
            "pocket_atoms",
            "pocket_coordinates",
            self.args.max_pocket_atoms,
        )

        apo_dataset = NormalizeDataset(dataset, "pocket_coordinates")

        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
        len_dataset = LengthDataset(src_pocket_dataset)
        src_pocket_dataset = TokenizeDataset(
            src_pocket_dataset,
            self.pocket_dictionary,
            max_seq_len=self.args.max_seq_len,
        )
        coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")
        src_pocket_dataset = PrependAndAppend(
            src_pocket_dataset,
            self.pocket_dictionary.bos(),
            self.pocket_dictionary.eos(),
        )
        pocket_edge_type = EdgeTypeDataset(
            src_pocket_dataset, len(self.pocket_dictionary)
        )
        coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
        distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
        coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
        distance_pocket_dataset = PrependAndAppend2DDataset(
            distance_pocket_dataset, 0.0
        )

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "pocket_src_tokens": RightPadDataset(
                        src_pocket_dataset,
                        pad_idx=self.pocket_dictionary.pad(),
                    ),
                    "pocket_src_distance": RightPadDataset2D(
                        distance_pocket_dataset,
                        pad_idx=0,
                    ),
                    "pocket_src_edge_type": RightPadDataset2D(
                        pocket_edge_type,
                        pad_idx=0,
                    ),
                    "pocket_src_coord": RightPadDatasetCoord(
                        coord_pocket_dataset,
                        pad_idx=0,
                    ),
                },
                "pocket_name": RawArrayDataset(poc_dataset),
                "pocket_len": RawArrayDataset(len_dataset),
            },
        )
        return nest_dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)

        if args.finetune_mol_model is not None:
            print("load pretrain model weight from...", args.finetune_mol_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_mol_model,
            )
            model.mol_model.load_state_dict(state["model"], strict=False)

        if args.finetune_pocket_model is not None:
            print("load pretrain model weight from...", args.finetune_pocket_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.finetune_pocket_model,
            )
            model.pocket_model.load_state_dict(state["model"], strict=False)

        if args.restore_model is not None:
            print("load pretrain model weight from...", args.restore_model)
            state = checkpoint_utils.load_checkpoint_to_cpu(
                args.restore_model,
            )
            model.load_state_dict(state["model"], strict=False)

        return model

    def train_step(
        self, sample, model, loss, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *loss*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~unicore.data.UnicoreDataset`.
            model (~unicore.models.BaseUnicoreModel): the model
            loss (~unicore.losses.UnicoreLoss): the loss
            optimizer (~unicore.optim.UnicoreOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """

        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = loss(model, sample)
        if ignore_grad:
            loss *= 0
        # print("loss: ", loss)
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, loss, test=False):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = loss(model, sample)
        return loss, sample_size, logging_output
