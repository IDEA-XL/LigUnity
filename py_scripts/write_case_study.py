import argparse
import gzip
import json
import multiprocessing as mp
import os
import pickle
import random

import lmdb
import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem.AllChem as AllChem
import torch
import tqdm
from biopandas.mol2 import PandasMol2
from biopandas.pdb import PandasPdb
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog('rdApp.*')

def gen_conformation(mol, num_conf=20, num_worker=8):
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, numThreads=num_worker, pruneRmsThresh=1, maxAttempts=10000, useRandomCoords=False)
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=num_worker)
        except:
            pass
        mol = Chem.RemoveHs(mol)
    except:
        print("cannot gen conf", Chem.MolToSmiles(mol))
        return None
    if mol.GetNumConformers() == 0:
        print("cannot gen conf", Chem.MolToSmiles(mol))
        return None
    return mol

def convert_2Dmol_to_data(smi, num_conf=1, num_worker=5):
    #to 3D
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = gen_conformation(mol, num_conf, num_worker)
    if mol is None:
        return None
    coords = [np.array(mol.GetConformer(i).GetPositions()) for i in range(mol.GetNumConformers())]
    atom_types = [a.GetSymbol() for a in mol.GetAtoms()]
    return {'coords': coords, 'atom_types': atom_types, 'smi': smi, 'mol': mol}

def convert_3Dmol_to_data(mol):

    if mol is None:
        return None
    coords = [np.array(mol.GetConformer(i).GetPositions()) for i in range(mol.GetNumConformers())]
    atom_types = [a.GetSymbol() for a in mol.GetAtoms()]
    return {'coords': coords, 'atom_types': atom_types, 'smi': Chem.MolToSmiles(mol), 'mol': mol}

def read_pdb(path):
    pdb_df = PandasPdb().read_pdb(path)

    coord = pdb_df.df['ATOM'][['x_coord', 'y_coord', 'z_coord']]
    atom_type = pdb_df.df['ATOM']['atom_name']
    residue_name = pdb_df.df['ATOM']['chain_id'] + pdb_df.df['ATOM']['residue_number'].astype(str)
    residue_type = pdb_df.df['ATOM']['residue_name']
    protein = {'coord': np.array(coord), 
               'atom_type': list(atom_type),
               'residue_name': list(residue_name),
               'residue_type': list(residue_type)}
    return protein


def read_sdf_gz_3d(path):
    inf = gzip.open(path)
    with Chem.ForwardSDMolSupplier(inf, removeHs=False, sanitize=False) as gzsuppl:
        ms = [add_charges(x) for x in gzsuppl if x is not None]
    ms = [rdMolStandardize.Uncharger().uncharge(Chem.RemoveHs(m)) for m in ms if m is not None]
    return ms

def add_charges(m):
    m.UpdatePropertyCache(strict=False)
    ps = Chem.DetectChemistryProblems(m)
    if not ps:
        Chem.SanitizeMol(m)
        return m
    for p in ps:
        if p.GetType()=='AtomValenceException':
            at = m.GetAtomWithIdx(p.GetAtomIdx())
            if at.GetAtomicNum()==7 and at.GetFormalCharge()==0 and at.GetExplicitValence()==4:
                at.SetFormalCharge(1)
            if at.GetAtomicNum()==6 and at.GetExplicitValence()==5:
                #remove a bond
                for b in at.GetBonds():
                    if b.GetBondType()==Chem.rdchem.BondType.DOUBLE:
                        b.SetBondType(Chem.rdchem.BondType.SINGLE)
                        break
            if at.GetAtomicNum()==8 and at.GetFormalCharge()==0 and at.GetExplicitValence()==3:
                at.SetFormalCharge(1)
            if at.GetAtomicNum()==5 and at.GetFormalCharge()==0 and at.GetExplicitValence()==4:
                at.SetFormalCharge(-1)
    try:
        Chem.SanitizeMol(m)
    except:
        return None
    return m

def get_different_raid(protein, ligand, raid=6):
    protein_coord = protein['coord']
    ligand_coord = ligand['coord']
    protein_residue_name = protein['residue_name']
    pocket_residue = set()
    for i in range(len(protein_coord)):
        for j in range(len(ligand_coord)):
            if np.linalg.norm(protein_coord[i] - ligand_coord[j]) < raid:
                pocket_residue.add(protein_residue_name[i])
    return pocket_residue

def read_mol2_ligand(path):
    mol2_df = PandasMol2().read_mol2(path)
    coord = mol2_df.df[['x', 'y', 'z']]
    atom_type = mol2_df.df['atom_name']
    ligand = {'coord': np.array(coord), 'atom_type': list(atom_type), 'mol': Chem.MolFromMol2File(path)}
    return ligand

def read_smi_mol(path):
    with open(path, 'r') as f:
        mols_lines = list(f.readlines())
    smis = [l.split(' ')[0] for l in mols_lines]
    mols = [Chem.MolFromSmiles(m) for m in smis]
    return mols

def parser(protein_path, mol_path, ligand_path, activity, pocket_index, raid=6):
    protein = read_pdb(protein_path)
    data_mols = read_smi_mol(mol_path)

    ligand = read_mol2_ligand(ligand_path)
    pocket_residue = get_different_raid(protein, ligand, raid=raid)
    pocket_atom_idx = [i for i, r in enumerate(protein['residue_name']) if r in pocket_residue]
    pocket_atom_type = [protein['atom_type'][i] for i in pocket_atom_idx]
    pocket_coord = [protein['coord'][i] for i in pocket_atom_idx]
    pocket_residue_type = [protein['residue_type'][i] for i in pocket_atom_idx]
    pocket_name = protein_path.split('/')[-2]
    pool = mp.Pool(32)
    #mols = [convert_2Dmol_to_data(m) for m in data_mols if m is not None]
    data_mols = [m for m in data_mols if m is not None]
    mols = [m for m in pool.imap_unordered(convert_2Dmol_to_data, data_mols)]
    mols = [m for m in mols if m is not None]
    
    return [{'atoms': m['atom_types'], 
            'coordinates': m['coords'], 
            'smi': m['smi'],
            'mol': ligand,
            'pocket_name': pocket_name,
            'pocket_index': pocket_index,
            'activity': activity, 
            "pocket_atom_type": pocket_atom_type, 
            "pocket_coord": pocket_coord} for m in mols]

def mol_parser(ligand_smis):
    pool = mp.Pool(16)
    mols = [m for m in pool.imap_unordered(convert_2Dmol_to_data, tqdm.tqdm(ligand_smis))]
    mols = [m for m in mols if m is not None]
    return [{'atoms': m['atom_types'],
            'coordinates': m['coords'],
            'smi': m['smi'],
            'mol': m['mol'],
            'label': 1,
            } for m in mols]

def pocket_parser(protein_path, ligand_path, pocket_index, pocket_name, raid=6):
    protein = read_pdb(protein_path)
    ligand = read_mol2_ligand(ligand_path)
    pocket_residue = get_different_raid(protein, ligand, raid=raid)
    pocket_atom_idx = [i for i, r in enumerate(protein['residue_name']) if r in pocket_residue]
    pocket_atom_type = [protein['atom_type'][i] for i in pocket_atom_idx]
    pocket_coord = [protein['coord'][i] for i in pocket_atom_idx]
    pocket_residue_type = [protein['residue_type'][i] for i in pocket_atom_idx]
    pocket_residue_name = [protein['residue_name'][i] for i in pocket_atom_idx]
    return {'pocket': pocket_name,
            'pocket_index': pocket_index,
            "pocket_atoms": pocket_atom_type, 
            "pocket_coordinates": pocket_coord,
            "pocket_residue_type": pocket_residue_type,
            "pocket_residue_name": pocket_residue_name}

def write_lmdb(data, lmdb_path):
    #resume
    if os.path.exists(lmdb_path):
        os.system(f"rm {lmdb_path}")
    env = lmdb.open(lmdb_path, subdir=False, readonly=False, lock=False, readahead=False, meminit=False, map_size=1099511627776)
    num = 0
    with env.begin(write=True) as txn:
        for d in data:
            txn.put(str(num).encode('ascii'), pickle.dumps(d))
            num += 1

import sys
if __name__ == '__main__':
    mode = sys.argv[1]

    if mode == 'mol':
        lig_file = sys.argv[2]
        lig_write_file = sys.argv[3]

        # read the ligands smiles into a list
        smis = json.load(open(lig_file))
        data = []
        print("number if ligands", len(set(smis)))
        d_active = (mol_parser(list(set(smis))))
        data.extend(d_active)

        # write ligands lmdb
        write_lmdb(data, lig_write_file)
    elif mode == 'pocket':
        prot_file = sys.argv[2]
        crystal_lig_file = sys.argv[3] # must be .mol2 file
        prot_write_file = sys.argv[4]

        # write pocket
        data = []
        d = pocket_parser(prot_file, crystal_lig_file, 1, "demo")
        data.append(d)
        write_lmdb(data, prot_write_file)

        

