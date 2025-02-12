# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json

# from IPython import embed as debug_embedded
import logging
import os
# from collections.abc import Iterable
# from sklearn.metrics import roc_auc_score
from xmlrpc.client import Boolean
import numpy as np
import torch
import pickle
from tqdm import tqdm
from unicore import checkpoint_utils
import unicore
from unicore.data import (AppendTokenDataset, Dictionary, EpochShuffleDataset,
                          FromNumpyDataset, NestedDictionaryDataset,
                          PrependTokenDataset, RawArrayDataset, LMDBDataset, RawLabelDataset,
                          RightPadDataset, RightPadDataset2D, TokenizeDataset, SortDataset, data_utils)
from unicore.tasks import UnicoreTask, register_task
from unimol.data import (AffinityDataset, CroppingPocketDataset,
                         CrossDistanceDataset, DistanceDataset,
                         EdgeTypeDataset, KeyDataset, LengthDataset,
                         NormalizeDataset, NormalizeDockingPoseDataset,
                         PrependAndAppend2DDataset, RemoveHydrogenDataset,
                         RemoveHydrogenPocketDataset, RightPadDatasetCoord,
                         RightPadDatasetCross2D, TTADockingPoseDataset, AffinityTestDataset, AffinityValidDataset,
                         AffinityMolDataset, AffinityPocketDataset, ResamplingDataset)
# from skchem.metrics import bedroc_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment
from sklearn.metrics import roc_curve

logger = logging.getLogger(__name__)
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def re_new(y_true, y_score, ratio):
    fp = 0
    tp = 0
    p = sum(y_true)
    n = len(y_true) - p
    num = ratio * n
    sort_index = np.argsort(y_score)[::-1]
    for i in range(len(sort_index)):
        index = sort_index[i]
        if y_true[index] == 1:
            tp += 1
        else:
            fp += 1
            if fp >= num:
                break
    return (tp * n) / (p * fp)


def calc_re(y_true, y_score, ratio_list):
    res2 = {}

    for ratio in ratio_list:
        res2[str(ratio)] = re_new(y_true, y_score, ratio)

    return res2


def cal_metrics(y_true, y_score, alpha):
    """
    Calculate BEDROC score.

    Parameters:
    - y_true: true binary labels (0 or 1)
    - y_score: predicted scores or probabilities
    - alpha: parameter controlling the degree of early retrieval emphasis

    Returns:
    - BEDROC score
    """

    # concate res_single and labels
    scores = np.expand_dims(y_score, axis=1)
    y_true = np.expand_dims(y_true, axis=1)
    scores = np.concatenate((scores, y_true), axis=1)
    # inverse sort scores based on first column
    scores = scores[scores[:, 0].argsort()[::-1]]
    bedroc = CalcBEDROC(scores, 1, 80.5)
    count = 0
    # sort y_score, return index
    index = np.argsort(y_score)[::-1]
    for i in range(int(len(index) * 0.005)):
        if y_true[index[i]] == 1:
            count += 1
    auc = CalcAUC(scores, 1)
    ef_list = CalcEnrichment(scores, 1, [0.005, 0.01, 0.02, 0.05])
    ef = {
        "0.005": ef_list[0],
        "0.01": ef_list[1],
        "0.02": ef_list[2],
        "0.05": ef_list[3]
    }
    re_list = calc_re(y_true, y_score, [0.005, 0.01, 0.02, 0.05])
    return auc, bedroc, ef, re_list


def get_uniprot_seq(uniprot):
    import urllib
    if not os.path.exists(f"./uniport_fasta/{uniprot}.fasta"):
        os.system(f"mkdir -p ./uniport_fasta")
        urllib.request.urlretrieve(f"https://rest.uniprot.org/uniprotkb/{uniprot}.fasta",
                                   f"./uniport_fasta/{uniprot}.fasta")

    with open(f"./uniport_fasta/{uniprot}.fasta", "r") as f:
        lines = []
        for line in f.readlines():
            if line.startswith(">"):
                continue
            else:
                lines.append(line.strip())
        return "".join(lines)


@register_task("test_task")
class ContrasRankTest(UnicoreTask):
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
            "--demo-uniprot",
            type=str,
            default=""
        )
        parser.add_argument("--reg", action="store_true", help="regression task")

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

    @classmethod
    def setup_task(cls, args, **kwargs):
        mol_dictionary = Dictionary.load(os.path.join(PROJECT_ROOT, "vocab", "dict_mol.txt"))
        pocket_dictionary = Dictionary.load(os.path.join(PROJECT_ROOT, "vocab", "dict_pkt.txt"))
        logger.info("ligand dictionary: {} types".format(len(mol_dictionary)))
        logger.info("pocket dictionary: {} types".format(len(pocket_dictionary)))
        return cls(args, mol_dictionary, pocket_dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        'smi','pocket','atoms','coordinates','pocket_atoms','pocket_coordinates'
        Args:
            split (str): name of the data scoure (e.g., bppp)
        """
        if split == "test":
            data_path = f"{self.args.data}/casf.lmdb"
        else:
            data_path = os.path.join(self.args.data, split + ".lmdb")
        dataset = LMDBDataset(data_path)
        if split.startswith("train"):
            smi_dataset = KeyDataset(dataset, "smi")
            poc_dataset = KeyDataset(dataset, "pocket")

            dataset = AffinityDataset(
                dataset,
                self.args.seed,
                "atoms",
                "coordinates",
                "pocket_atoms",
                "pocket_coordinates",
                "label",
                True,
            )
            tgt_dataset = KeyDataset(dataset, "affinity")

        else:

            dataset = AffinityDataset(
                dataset,
                self.args.seed,
                "atoms",
                "coordinates",
                "pocket_atoms",
                "pocket_coordinates",
                "label",
            )
            tgt_dataset = KeyDataset(dataset, "affinity")
            smi_dataset = KeyDataset(dataset, "smi")
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

        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)

        apo_dataset = NormalizeDataset(dataset, "coordinates")
        apo_dataset = NormalizeDataset(apo_dataset, "pocket_coordinates")

        src_dataset = KeyDataset(apo_dataset, "atoms")
        mol_len_dataset = LengthDataset(src_dataset)
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

        src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
        pocket_len_dataset = LengthDataset(src_pocket_dataset)
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
                    "mol_len": RawArrayDataset(mol_len_dataset),
                    "pocket_len": RawArrayDataset(pocket_len_dataset)
                },
                "target": {
                    "finetune_target": RawLabelDataset(tgt_dataset),
                },
                "smi_name": RawArrayDataset(smi_dataset),
                "pocket_name": RawArrayDataset(poc_dataset),
            },
        )
        if split == "train" and kwargs.get("shuffle", True):
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(src_dataset))

            self.datasets[split] = SortDataset(
                nest_dataset,
                sort_order=[shuffle],
            )
            self.datasets[split] = ResamplingDataset(
                self.datasets[split]
            )
        else:
            self.datasets[split] = nest_dataset

        return self.datasets[split]

    def load_mols_dataset(self, data_path, atoms, coords, **kwargs):

        dataset = LMDBDataset(data_path)
        # label_dataset = KeyDataset(dataset, "label")
        try:
            label_dataset = KeyDataset(dataset, "label")
            x = label_dataset[0]
        except:
            label_dataset = None
        dataset = AffinityMolDataset(
            dataset,
            self.args.seed,
            atoms,
            coords,
            False,
        )

        smi_dataset = KeyDataset(dataset, "smi")
        mol_dataset = KeyDataset(dataset, "mol")
        if kwargs.get("load_name", False):
            name_dataset = KeyDataset(dataset, "name")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)
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

        if label_dataset is not None:
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
                "target": RawArrayDataset(label_dataset),
                "mol_len": RawArrayDataset(len_dataset),
                "mol": RawArrayDataset(mol_dataset)
            }
        else:
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
                # "target": RawArrayDataset(label_dataset),
                "mol_len": RawArrayDataset(len_dataset),
                "mol": RawArrayDataset(mol_dataset)
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
            False,
            "pocket"
        )
        poc_dataset = KeyDataset(dataset, "pocket")
        resname_dataset = KeyDataset(dataset, "pocket_residue_name")

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
                "pocket_residue_names": RawArrayDataset(resname_dataset),
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

        return model

    def test_pcba_target(self, name, model, seq, **kwargs):
        """Encode a dataset with the molecule encoder."""

        # names = "PPARG"
        data_path = f"{self.args.data}/lit_pcba/" + name + "/mols.lmdb"
        mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
        num_data = len(mol_dataset)
        bsz = self.args.batch_size
        # print(num_data//bsz)
        mol_reps = []
        mol_names = []
        labels = []

        # generate mol data

        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, num_workers=8,
                                               collate_fn=mol_dataset.collater)
        for _, sample in enumerate(tqdm(mol_data)):
            sample = unicore.utils.move_to_cuda(sample)
            mol_emb = model.mol_forward(**sample["net_input"])
            mol_emb = mol_emb.detach().cpu().numpy()
            mol_reps.append(mol_emb)
            mol_names.extend(sample["smi_name"])
            labels.extend(sample["target"].detach().cpu().numpy())
        mol_reps = np.concatenate(mol_reps, axis=0)
        labels = np.array(labels, dtype=np.int32)
        # generate pocket data
        data_path = f"{self.args.data}/lit_pcba/" + name + "/pockets.lmdb"
        pocket_dataset = self.load_pockets_dataset(data_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=bsz, collate_fn=pocket_dataset.collater)
        pocket_reps = []
        pocket_names = []

        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            pocket_emb = model.pocket_forward(protein_sequences=seq, **sample["net_input"])
            pocket_emb = pocket_emb.detach().cpu().numpy()
            pocket_name = sample["pocket_name"]
            pocket_names.append(pocket_name)
            pocket_reps.append(pocket_emb)
        pocket_reps = np.concatenate(pocket_reps, axis=0)

        os.system(f"mkdir -p {self.args.results_path}/PCBA/{name}")
        np.save(f"{self.args.results_path}/PCBA/{name}/saved_mols_embed.npy", mol_reps)
        np.save(f"{self.args.results_path}/PCBA/{name}/saved_target_embed.npy", pocket_reps)
        np.save(f"{self.args.results_path}/PCBA/{name}/saved_labels.npy", labels)
        json.dump(pocket_names, open(f"{self.args.results_path}/PCBA/{name}/saved_pocket_names.json", "w"))

        res = pocket_reps @ mol_reps.T
        res_single = res.max(axis=0)
        auc, bedroc, ef_list, re_list = cal_metrics(labels, res_single, 80.5)

        return auc, bedroc, ef_list, re_list

    def test_pcba_target_regression(self, name, model, seq, **kwargs):
        """Encode a dataset with the molecule encoder."""

        # names = "PPARG"
        data_path = f"{self.args.data}/lit_pcba/" + name + "/mols.lmdb"
        mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
        num_data = len(mol_dataset)
        bsz = self.args.batch_size
        # print(num_data//bsz)
        mol_names = []
        labels = []
        act_preds_all = []

        # generate mol data

        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater,
                                               num_workers=8)
        for _, mol_sample in enumerate(tqdm(mol_data)):
            mol_sample = unicore.utils.move_to_cuda(mol_sample)
            mol_names.extend(mol_sample["smi_name"])
            labels.extend(mol_sample["target"].detach().cpu().numpy())

            # generate pocket data
            data_path = f"{self.args.data}/lit_pcba/" + name + "/pockets.lmdb"
            pocket_dataset = self.load_pockets_dataset(data_path)
            pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=bsz,
                                                      collate_fn=pocket_dataset.collater)
            act_preds = []
            pocket_names = []

            for _, pocket_sample in enumerate(pocket_data):
                pocket_sample = unicore.utils.move_to_cuda(pocket_sample)
                pred = model.forward(protein_sequences=seq, **pocket_sample["net_input"], **mol_sample["net_input"])
                pocket_name = pocket_sample["pocket_name"]
                act_preds.append(pred.detach().cpu().numpy())
                pocket_names.append(pocket_name)

            act_preds = np.concatenate(act_preds, axis=0)  # [num_pocket, num_lig]
            act_preds_all.append(act_preds)

        labels = np.array(labels, dtype=np.int32)
        res = np.concatenate(act_preds_all, axis=1)
        res_single = res.max(axis=0)
        os.system(f"mkdir -p {self.args.results_path}/PCBA/{name}")
        np.save(f"{self.args.results_path}/PCBA/{name}/saved_labels.npy", labels)
        np.save(f"{self.args.results_path}/PCBA/{name}/saved_preds.npy", res_single)
        json.dump(pocket_names, open(f"{self.args.results_path}/PCBA/{name}/saved_pocket_names.json", "w"))

        auc, bedroc, ef_list, re_list = cal_metrics(labels, res_single, 80.5)

        return auc, bedroc, ef_list, re_list

    def test_pcba(self, model, **kwargs):
        targets = os.listdir(f"{self.args.data}/lit_pcba/")

        # print(targets)
        auc_list = []
        ef_list = []
        bedroc_list = []

        re_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": []
        }
        ef_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": []
        }
        uniprot_list = json.load(open(f"{self.args.data}/PCBA.json"))
        target2uniport = {x[2]: x[0] for x in uniprot_list}

        for target in targets:
            print(target)
            # if os.path.exists(f"{self.args.results_path}/PCBA/{target}/saved_labels.npy"):
            #     continue
            seq = get_uniprot_seq(target2uniport[target])
            if self.args.arch in ["DTA", "pocketregression"]:
                auc, bedroc, ef, re = self.test_pcba_target_regression(target, model, seq)
            else:
                auc, bedroc, ef, re = self.test_pcba_target(target, model, seq)
            auc_list.append(auc)
            bedroc_list.append(bedroc)
            for key in ef:
                ef_list[key].append(ef[key])
            print("re", re)
            print("ef", ef)
            for key in re:
                re_list[key].append(re[key])
        print(auc_list)
        print(ef_list)
        print("auc 25%", np.percentile(auc_list, 25))
        print("auc 50%", np.percentile(auc_list, 50))
        print("auc 75%", np.percentile(auc_list, 75))
        print("auc mean", np.mean(auc_list))
        print("bedroc 25%", np.percentile(bedroc_list, 25))
        print("bedroc 50%", np.percentile(bedroc_list, 50))
        print("bedroc 75%", np.percentile(bedroc_list, 75))
        print("bedroc mean", np.mean(bedroc_list))
        # print(np.median(auc_list))
        # print(np.median(ef_list))
        for key in ef_list:
            print("ef", key, "25%", np.percentile(ef_list[key], 25))
            print("ef", key, "50%", np.percentile(ef_list[key], 50))
            print("ef", key, "75%", np.percentile(ef_list[key], 75))
            print("ef", key, "mean", np.mean(ef_list[key]))
        for key in re_list:
            print("re", key, "25%", np.percentile(re_list[key], 25))
            print("re", key, "50%", np.percentile(re_list[key], 50))
            print("re", key, "75%", np.percentile(re_list[key], 75))
            print("re", key, "mean", np.mean(re_list[key]))

        return

    def test_dude_target(self, target, model, seq, **kwargs):

        data_path = f"{self.args.data}/DUD-E/" + target + "/mols_real.lmdb"
        mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
        num_data = len(mol_dataset)
        bsz = 64
        print(num_data // bsz)
        mol_reps = []
        mol_names = []
        labels = []

        # generate mol data
        print("begin with target:", target)
        print("number of mol:", len(mol_dataset))
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, num_workers=8,
                                               collate_fn=mol_dataset.collater)

        for _, sample in enumerate(tqdm(mol_data)):
            sample = unicore.utils.move_to_cuda(sample)
            mol_emb = model.mol_forward(**sample["net_input"])
            mol_emb = mol_emb.detach().cpu().numpy()
            # print(mol_emb.dtype)
            mol_reps.append(mol_emb)
            mol_names.extend(sample["smi_name"])
            labels.extend(sample["target"].detach().cpu().numpy())
        mol_reps = np.concatenate(mol_reps, axis=0)
        labels = np.array(labels, dtype=np.int32)
        # generate pocket data
        data_path = f"{self.args.data}/DUD-E/" + target + "/pocket.lmdb"
        pocket_dataset = self.load_pockets_dataset(data_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=bsz, collate_fn=pocket_dataset.collater)
        pocket_reps = []

        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            pocket_emb = model.pocket_forward(protein_sequences=seq, **sample["net_input"])
            pocket_emb = pocket_emb.detach().cpu().numpy()
            pocket_reps.append(pocket_emb)
        pocket_reps = np.concatenate(pocket_reps, axis=0)
        print(pocket_reps.shape)
        res = pocket_reps @ mol_reps.T

        res_single = res.max(axis=0)
        os.system(f"mkdir -p {self.args.results_path}/DUDE/{target}")
        np.save(f"{self.args.results_path}/DUDE/{target}/saved_mols_embed.npy", mol_reps)
        np.save(f"{self.args.results_path}/DUDE/{target}/saved_target_embed.npy", pocket_reps)
        np.save(f"{self.args.results_path}/DUDE/{target}/saved_labels.npy", labels)
        auc, bedroc, ef_list, re_list = cal_metrics(labels, res_single, 80.5)

        print(target)
        print("ef:", ef_list)

        return auc, bedroc, ef_list, re_list, res_single, labels

    def test_dude_target_regression(self, target, model, seq, **kwargs):

        data_path = f"{self.args.data}/DUD-E/" + target + "/mols_real.lmdb"
        mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
        num_data = len(mol_dataset)
        bsz = 64
        print(num_data // bsz)
        mol_reps = []
        mol_names = []
        labels = []

        # generate mol data
        print("begin with target:", target)
        print("number of mol:", len(mol_dataset))
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)

        act_preds_all = []

        # generate mol data

        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater,
                                               num_workers=8)
        for _, mol_sample in enumerate(tqdm(mol_data)):
            mol_sample = unicore.utils.move_to_cuda(mol_sample)
            mol_names.extend(mol_sample["smi_name"])
            labels.extend(mol_sample["target"].detach().cpu().numpy())

            # generate pocket data
            data_path = f"{self.args.data}/DUD-E/" + target + "/pocket.lmdb"
            pocket_dataset = self.load_pockets_dataset(data_path)
            pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=bsz,
                                                      collate_fn=pocket_dataset.collater)
            act_preds = []
            pocket_names = []

            for _, pocket_sample in enumerate(pocket_data):
                pocket_sample = unicore.utils.move_to_cuda(pocket_sample)
                pred = model.forward(protein_sequences=seq, **pocket_sample["net_input"], **mol_sample["net_input"])
                pocket_name = pocket_sample["pocket_name"]
                act_preds.append(pred.detach().cpu().numpy())
                pocket_names.append(pocket_name)

            act_preds = np.concatenate(act_preds, axis=0)  # [num_pocket, num_lig]
            act_preds_all.append(act_preds)

        res = np.concatenate(act_preds_all, axis=1)
        res_single = res.max(axis=0)
        os.system(f"mkdir -p {self.args.results_path}/DUDE/{target}")
        np.save(f"{self.args.results_path}/DUDE/{target}/saved_labels.npy", labels)
        np.save(f"{self.args.results_path}/DUDE/{target}/saved_preds.npy", res_single)
        auc, bedroc, ef_list, re_list = cal_metrics(labels, res_single, 80.5)

        print(target)
        print("ef:", ef_list)

        return auc, bedroc, ef_list, re_list, res_single, labels

    def test_dude(self, model, **kwargs):

        targets = list(os.listdir(f"{self.args.data}/DUD-E/"))
        auc_list = []
        bedroc_list = []
        ef_list = []
        res_list = []
        labels_list = []
        re_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": [],
        }
        ef_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": [],
        }
        targets.reverse()
        uniprot_list = json.load(open(f"{self.args.data}/dude.json"))
        target2uniport = {x[2]: x[0] for x in uniprot_list}

        for i, target in enumerate(targets):
            seq = get_uniprot_seq(target2uniport[target.upper()])
            if self.args.arch in ["DTA", "pocketregression"]:
                auc, bedroc, ef, re, res_single, labels = self.test_dude_target_regression(target, model, seq)
            else:
                auc, bedroc, ef, re, res_single, labels = self.test_dude_target(target, model, seq)
            auc_list.append(auc)
            bedroc_list.append(bedroc)
            for key in ef:
                ef_list[key].append(ef[key])
            for key in re_list:
                re_list[key].append(re[key])

        print("auc mean", np.mean(auc_list))
        print("bedroc mean", np.mean(bedroc_list))

        for key in ef_list:
            print("ef", key, "mean", np.mean(ef_list[key]))

        for key in re_list:
            print("re", key, "mean", np.mean(re_list[key]))

        return

    def test_dekois_target(self, target, model, seq, **kwargs):

        data_path = f"{self.args.data}/DEKOIS_2.0x/{target}/{target}_lig.lmdb"
        mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
        num_data = len(mol_dataset)
        bsz = 64
        print(num_data // bsz)
        mol_reps = []
        mol_names = []
        labels = []

        # generate mol data
        print("begin with target:", target)
        print("number of mol:", len(mol_dataset))
        mol_data = torch.utils.data.DataLoader(mol_dataset, num_workers=4, batch_size=bsz,
                                               collate_fn=mol_dataset.collater)

        for _, sample in enumerate(tqdm(mol_data)):
            sample = unicore.utils.move_to_cuda(sample)
            mol_emb = model.mol_forward(**sample["net_input"])
            mol_emb = mol_emb.detach().cpu().numpy()
            # print(mol_emb.dtype)
            mol_reps.append(mol_emb)
            mol_names.extend(sample["smi_name"])
            labels.extend(sample["target"].detach().cpu().numpy())
        mol_reps = np.concatenate(mol_reps, axis=0)
        labels = np.array(labels, dtype=np.int32)
        # generate pocket data
        data_path = f"{self.args.data}/DEKOIS_2.0x/{target}/{target}_pocket.lmdb"
        pocket_dataset = self.load_pockets_dataset(data_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=bsz, collate_fn=pocket_dataset.collater)
        pocket_reps = []

        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            pocket_emb = model.pocket_forward(protein_sequences=seq, **sample["net_input"])
            pocket_emb = pocket_emb.detach().cpu().numpy()
            pocket_reps.append(pocket_emb)
        pocket_reps = np.concatenate(pocket_reps, axis=0)
        print(pocket_reps.shape)
        res = pocket_reps @ mol_reps.T

        res_single = res.max(axis=0)
        os.system(f"mkdir -p {self.args.results_path}/DEKOIS/{target}")
        print(f"writing to {self.args.results_path}/DEKOIS/{target}")
        np.save(f"{self.args.results_path}/DEKOIS/{target}/saved_mols_embed.npy", mol_reps)
        np.save(f"{self.args.results_path}/DEKOIS/{target}/saved_target_embed.npy", pocket_reps)
        np.save(f"{self.args.results_path}/DEKOIS/{target}/saved_labels.npy", labels)
        auc, bedroc, ef_list, re_list = cal_metrics(labels, res_single, 80.5)

        print(target)
        print("ef:", ef_list)

        return auc, bedroc, ef_list, re_list, res_single, labels

    def test_dekois_target_regression(self, target, model, seq, **kwargs):

        data_path = f"{self.args.data}/DEKOIS_2.0x/{target}/{target}_lig.lmdb"
        mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
        num_data = len(mol_dataset)
        bsz = 64
        print(num_data // bsz)
        mol_reps = []
        mol_names = []
        labels = []
        act_preds_all = []

        # generate mol data
        print("begin with target:", target)
        print("number of mol:", len(mol_dataset))
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater,
                                               num_workers=8)
        for _, mol_sample in enumerate(tqdm(mol_data)):
            mol_sample = unicore.utils.move_to_cuda(mol_sample)
            mol_names.extend(mol_sample["smi_name"])
            labels.extend(mol_sample["target"].detach().cpu().numpy())

            # generate pocket data
            data_path = f"{self.args.data}/DEKOIS_2.0x/{target}/{target}_pocket.lmdb"
            pocket_dataset = self.load_pockets_dataset(data_path)
            pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=bsz,
                                                      collate_fn=pocket_dataset.collater)
            act_preds = []
            pocket_names = []

            for _, pocket_sample in enumerate(pocket_data):
                pocket_sample = unicore.utils.move_to_cuda(pocket_sample)
                pred = model.forward(protein_sequences=seq, **pocket_sample["net_input"], **mol_sample["net_input"])
                pocket_name = pocket_sample["pocket_name"]
                act_preds.append(pred.detach().cpu().numpy())
                pocket_names.append(pocket_name)

            act_preds = np.concatenate(act_preds, axis=0)  # [num_pocket, num_lig]
            act_preds_all.append(act_preds)

        res = np.concatenate(act_preds_all, axis=1)

        res_single = res.max(axis=0)
        os.system(f"mkdir -p {self.args.results_path}/DEKOIS/{target}")
        print(f"writing to {self.args.results_path}/DEKOIS/{target}")
        np.save(f"{self.args.results_path}/DEKOIS/{target}/saved_labels.npy", labels)
        np.save(f"{self.args.results_path}/DEKOIS/{target}/saved_preds.npy", res_single)
        auc, bedroc, ef_list, re_list = cal_metrics(labels, res_single, 80.5)

        print(target)
        print("ef:", ef_list)

        return auc, bedroc, ef_list, re_list, res_single, labels

    def test_dekois(self, model, **kwargs):

        targets = list(os.listdir(f"{self.args.data}/DEKOIS_2.0x/"))
        auc_list = []
        bedroc_list = []
        ef_list = []
        res_list = []
        labels_list = []
        re_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": [],
        }
        ef_list = {
            "0.005": [],
            "0.01": [],
            "0.02": [],
            "0.05": [],
        }
        targets.reverse()

        uniprot_list = json.load(open(f"{self.args.data}/dekois.json"))
        target2uniport = {x[2]: x[0] for x in uniprot_list}

        for i, target in enumerate(targets):
            if not os.path.exists(f"{self.args.data}/DEKOIS_2.0x/{target}/{target}_lig.lmdb"):
                continue
            seq = get_uniprot_seq(target2uniport[target.upper()])
            if self.args.arch in ["DTA", "pocketregression"]:
                auc, bedroc, ef, re, res_single, labels = self.test_dekois_target_regression(target, model, seq)
            else:
                auc, bedroc, ef, re, res_single, labels = self.test_dekois_target(target, model, seq)
            auc_list.append(auc)
            bedroc_list.append(bedroc)
            for key in ef:
                ef_list[key].append(ef[key])
            for key in re_list:
                re_list[key].append(re[key])
            # except Exception as e:
            #     print(target, e)
            #     continue

        print("auc mean", np.mean(auc_list))
        print("bedroc mean", np.mean(bedroc_list))

        for key in ef_list:
            print("ef", key, "mean", np.mean(ef_list[key]))

        for key in re_list:
            print("re", key, "mean", np.mean(re_list[key]))

        return

    def test_demo(self, model):
        data_path = self.args.demo_lig_file
        mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")

        bsz = 64
        mol_reps = []
        mol_smis = []
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)

        for _, sample in enumerate(mol_data):
            sample = unicore.utils.move_to_cuda(sample)
            mol_emb = model.mol_forward(**sample["net_input"])
            mol_emb = mol_emb.detach().cpu().numpy()
            # print(mol_emb.dtype)
            mol_reps.append(mol_emb)
            mol_smis.extend(sample["smi_name"])
        mol_reps = np.concatenate(mol_reps, axis=0)

        data_path = self.args.demo_prot_file
        pocket_dataset = self.load_pockets_dataset(data_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=bsz, collate_fn=pocket_dataset.collater)
        sample = list(pocket_data)[0]

        sample = unicore.utils.move_to_cuda(sample)
        seq = get_uniprot_seq(self.args.demo_uniprot)
        pocket_emb = model.pocket_forward(protein_sequences=seq, **sample["net_input"])
        pocket_reps = pocket_emb.detach().cpu().numpy()

        os.system(f"mkdir -p {self.args.results_path}")
        json.dump(mol_smis, open(f"{self.args.results_path}/saved_smis.json", "w"))
        np.save(f"{self.args.results_path}/saved_mols_embed.npy", mol_reps)
        np.save(f"{self.args.results_path}/saved_target_embed.npy", pocket_reps)

    def test_fep_target(self, target, model, label_info, **kwargs):
        data_path = f"{self.args.data}/FEP/lmdbs/{target}_lig.lmdb"
        mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
        num_data = len(mol_dataset)
        bsz = 64
        mol_reps = []
        mol_smis = []
        labels = []

        # generate mol data
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)

        for _, sample in enumerate(mol_data):
            sample = unicore.utils.move_to_cuda(sample)
            mol_emb = model.mol_forward(**sample["net_input"])
            mol_emb = mol_emb.detach().cpu().numpy()
            # print(mol_emb.dtype)
            mol_reps.append(mol_emb)
            mol_smis.extend(sample["smi_name"])
        mol_reps = np.concatenate(mol_reps, axis=0)
        # generate pocket data
        data_path = f"{self.args.data}/FEP/lmdbs/{target}.lmdb"
        pocket_dataset = self.load_pockets_dataset(data_path)
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=bsz, collate_fn=pocket_dataset.collater)
        pocket_reps = []

        for _, sample in enumerate(pocket_data):
            sample = unicore.utils.move_to_cuda(sample)
            seq = label_info["sequence"]
            pocket_emb = model.pocket_forward(protein_sequences=seq, **sample["net_input"])
            pocket_emb = pocket_emb.detach().cpu().numpy()
            pocket_reps.append(pocket_emb)

        pocket_reps = np.concatenate(pocket_reps, axis=0)
        res = pocket_reps @ mol_reps.T

        res_single = res.max(axis=0)
        act_dict = {}
        for lig in label_info["ligands"]:
            act_dict[lig["smi"]] = float(lig["act"])
        real_dg = np.array([act_dict[smi] for smi in mol_smis])
        pred_dg = res_single
        from scipy import stats
        corr = stats.pearsonr(real_dg, pred_dg).statistic
        if corr < 0:
            r2 = 0
        else:
            r2 = corr ** 2

        os.system(f"mkdir -p {self.args.results_path}/FEP/{target}")
        np.save(f"{self.args.results_path}/FEP/{target}/saved_mols_embed.npy", mol_reps)
        np.save(f"{self.args.results_path}/FEP/{target}/saved_target_embed.npy", pocket_reps)
        np.save(f"{self.args.results_path}/FEP/{target}/saved_labels.npy", real_dg)
        json.dump(mol_smis, open(f"{self.args.results_path}/FEP/{target}/saved_smis.json", "w"))
        return r2

    def test_fep_target_regression(self, target, model, label_info, **kwargs):

        data_path = f"{self.args.data}/FEP/lmdbs/{target}_lig.lmdb"
        mol_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
        num_data = len(mol_dataset)
        bsz = 64
        act_preds_all = []
        mol_smis = []
        labels = []

        # generate mol data
        mol_data = torch.utils.data.DataLoader(mol_dataset, batch_size=bsz, collate_fn=mol_dataset.collater)
        for _, mol_sample in enumerate(tqdm(mol_data)):
            mol_sample = unicore.utils.move_to_cuda(mol_sample)
            mol_smis.extend(mol_sample["smi_name"])
            labels.extend(mol_sample["target"].detach().cpu().numpy())

            # generate pocket data
            data_path = f"{self.args.data}/FEP/lmdbs/{target}.lmdb"
            pocket_dataset = self.load_pockets_dataset(data_path)
            pocket_data = torch.utils.data.DataLoader(pocket_dataset, batch_size=bsz,
                                                      collate_fn=pocket_dataset.collater)
            act_preds = []
            pocket_names = []

            for _, pocket_sample in enumerate(tqdm(pocket_data)):
                pocket_sample = unicore.utils.move_to_cuda(pocket_sample)
                seq = label_info["sequence"]
                pred = model.forward(protein_sequences=seq, **pocket_sample["net_input"], **mol_sample["net_input"])
                pocket_name = pocket_sample["pocket_name"]
                pocket_names.append(pocket_name)
                print("pred", pred.shape)
                act_preds.append(pred.detach().cpu().numpy() + 6.)

            act_preds = np.concatenate(act_preds, axis=0)  # [num_pocket, num_lig]
            print("act_preds", act_preds.shape)
            act_preds_all.append(act_preds)

        res = np.concatenate(act_preds_all, axis=1)
        res_single = res.max(axis=0)
        act_dict = {}
        for lig in label_info["ligands"]:
            act_dict[lig["smi"]] = float(lig["act"])
        real_dg = np.array([act_dict[smi] for smi in mol_smis])
        pred_dg = res_single
        from scipy import stats
        corr = stats.pearsonr(real_dg, pred_dg).statistic
        # if corr < 0:
        #     r2 = 0
        # else:
        #     r2 = corr ** 2

        os.system(f"mkdir -p {self.args.results_path}/FEP/{target}")
        np.save(f"{self.args.results_path}/FEP/{target}/saved_labels.npy", real_dg)
        np.save(f"{self.args.results_path}/FEP/{target}/saved_preds.npy", pred_dg)
        json.dump(mol_smis, open(f"{self.args.results_path}/FEP/{target}/saved_smis.json", "w"))

        return corr

    def test_fep(self, model, **kwargs):
        labels_fep = json.load(
            open(f"{self.args.data}/FEP/fep_labels.json"))
        ligands_dict = {x["pockets"][0]: x for x in labels_fep}
        rho_list = []
        for i, target in enumerate(ligands_dict.keys()):
            if self.args.arch in ["DTA", "pocketregression"]:
                rho = self.test_fep_target_regression(target, model, ligands_dict[target])
            else:
                rho = self.test_fep_target(target, model, ligands_dict[target])
            # print(target, rho)
            rho_list.append(rho)

        print(self.args.results_path.split("/")[-1], np.mean(rho_list), np.median(rho_list))

    def inference_pdbbind(self, model, split="train", **kwargs):
        pdbbind_dataset = self.load_dataset(split, load_name=True, shuffle=False)
        num_data = len(pdbbind_dataset)
        bsz = 32
        print(num_data // bsz)
        mol_reps = []
        pocket_reps = []
        pdbbind_ids = []
        mol_smis = []
        pocket2seq = {}
        if split == "train":
            pdbbind_label = json.load(open(f"{self.args.data}/train_label_pdbbind_seq.json"))
        else:
            pdbbind_label = json.load(open(f"/casf_label_seq.json"))
        for assay in pdbbind_label:
            seq = assay["sequence"]
            pockets = assay["pockets"]
            for pocket in pockets:
                pocket2seq[pocket.split("_")[0]] = seq

        # generate mol data
        print("number of data:", len(pdbbind_dataset))
        mol_data = torch.utils.data.DataLoader(pdbbind_dataset, num_workers=8, batch_size=bsz,
                                               collate_fn=pdbbind_dataset.collater)

        for _, sample in enumerate(tqdm(mol_data)):
            # compute molecular embedding
            sample = unicore.utils.move_to_cuda(sample)
            pocket_names = sample["pocket_name"]
            seq = [pocket2seq[x] for x in pocket_names]
            mol_emb, pocket_emb, _, _ = model.forward(**sample["net_input"], protein_sequences=seq)
            mol_emb = mol_emb[0].detach().cpu().numpy()
            mol_reps.append(mol_emb)
            pocket_emb = pocket_emb[0].detach().cpu().numpy()
            pocket_reps.append(pocket_emb)

            pdbbind_ids += sample["pocket_name"]
            mol_smis += sample["smi_name"]

        mol_reps = np.concatenate(mol_reps, axis=0)
        pocket_reps = np.concatenate(pocket_reps, axis=0)

        write_dir = f"{self.args.results_path}/PDBBind"
        if not os.path.exists(write_dir):
            os.system(f"mkdir -p {write_dir}")
        np.save(f"{write_dir}/{split}_mol_reps.npy", mol_reps)
        np.save(f"{write_dir}/{split}_pocket_reps.npy", pocket_reps)
        json.dump(pdbbind_ids, open(f"{write_dir}/{split}_pdbbind_ids.json", "w"))
        json.dump(mol_smis, open(f"{write_dir}/{split}_mol_smis.json", "w"))

    def inference_bdb_lig(self, model):
        data_path = f"{self.args.data}/train_lig_all_blend.lmdb"
        bdb_dataset = self.load_mols_dataset(data_path, "atoms", "coordinates")
        num_data = len(bdb_dataset)
        bsz = 128
        print(num_data // bsz)
        # generate mol data
        print("number of data:", len(bdb_dataset))

        mol_reps = []
        mol_smis = []
        mol_data = torch.utils.data.DataLoader(bdb_dataset, num_workers=8, batch_size=bsz,
                                               collate_fn=bdb_dataset.collater)
        for _, sample in enumerate(tqdm(mol_data)):
            # compute molecular embedding
            sample = unicore.utils.move_to_cuda(sample)
            mol_emb = model.mol_forward(**sample["net_input"])
            mol_emb = mol_emb.detach().cpu().numpy()
            mol_reps.append(mol_emb)
            mol_smis += sample["smi_name"]

        mol_reps = np.concatenate(mol_reps, axis=0)

        write_dir = f"{self.args.results_path}/BDB"
        if not os.path.exists(write_dir):
            os.mkdir(write_dir)
        np.save(f"{write_dir}/bdb_mol_reps.npy", mol_reps)
        json.dump(mol_smis, open(f"{write_dir}/bdb_mol_smis.json", "w"))

    def inference_bdb_pocket(self, model):
        data_path = f"{self.args.data}/train_prot_all_blend.lmdb"
        blend_label = json.load(open(f"{self.args.data}/train_label_blend_seq_full.json"))
        pocket_dataset = self.load_pockets_dataset(data_path)
        bsz = 32
        pocket_data = torch.utils.data.DataLoader(pocket_dataset, num_workers=8, batch_size=bsz,
                                                  collate_fn=pocket_dataset.collater)
        pocket_reps = []
        pocket_names = []
        pocket2seq = {}
        for assay in blend_label:
            seq = assay["sequence"]
            pockets = assay["pockets"]
            for pocket in pockets:
                pocket2seq[pocket] = seq

        for _, sample in enumerate(tqdm(pocket_data)):
            sample = unicore.utils.move_to_cuda(sample)
            pocket_name = sample["pocket_name"]
            seq_list = [pocket2seq.get(x, "") for x in pocket_name]

            pocket_emb = model.pocket_forward(protein_sequences=seq_list, **sample["net_input"])
            pocket_emb = pocket_emb.detach().cpu().numpy()
            for seq, emb, name in zip(seq_list, pocket_emb, sample["pocket_name"]):
                if seq != "":
                    pocket_names.append(name)
                    pocket_reps.append(emb)
        pocket_reps = np.stack(pocket_reps, axis=0)
        print(pocket_reps.shape)
        write_dir = f"{self.args.results_path}/BDB"
        if not os.path.exists(write_dir):
            os.mkdir(write_dir)
        np.save(f"{write_dir}/bdb_pocket_reps.npy", pocket_reps)
        json.dump(pocket_names, open(f"{write_dir}/bdb_pocket_names.json", "w"))