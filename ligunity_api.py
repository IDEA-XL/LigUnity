import os
import json
import numpy as np
from typing import Union

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}

ckpt_path = "./ckpts/LigUnity_pocket_ranking"

@app.get("/predict")
async def read_item(input_json: Union[str, None] = None):

    input_data = json.load(open(input_json))
    os.makedirs(input_data["output_dir"], exist_ok=True)

    ligand_lmdb = os.path.join(input_data["output_dir"],"query_ligands.lmdb")
    pocket_lmdb = os.path.join(input_data["output_dir"],"pocket.lmdb")
    query_ligands = input_data["query_ligands"]
    protein_pdb = input_data["protein_pdb"]
    ref_ligand = input_data["ref_ligand"]
    os.system(f"python ./py_scripts/write_case_study.py mol {query_ligands} {ligand_lmdb}")
    os.system(f"python ./py_scripts/write_case_study.py pocket {protein_pdb} {ref_ligand} {pocket_lmdb}")

    uniprot = input_data["uniprot"]

    for i in range(5):
        path2result = os.path.join(input_data["output_dir"],f"res_{i+1}")
        path2weight = os.path.join(ckpt_path, f"checkpoint_avg_41-50_{i+1}.pt")
        running_cmd = f"bash test_zeroshot_demo.sh {ligand_lmdb} {pocket_lmdb} {uniprot} pocket_ranking {path2weight} {path2result}"

        os.system(running_cmd)
    
    pred_all = []
    for i in range(5):
        path2result = os.path.join(input_data["output_dir"],f"res_{i+1}")
        lig_embed_i = np.load(os.path.join(path2result,"saved_mols_embed.npy"))
        pocket_embed_i = np.load(os.path.join(path2result, "saved_target_embed.npy"))
        pred_i = pocket_embed_i @ lig_embed_i.T
        pred_all.append(pred_i.squeeze(0))

    pred_ensemble = np.stack(pred_all).mean(axis=0)
    smis = json.load(open(os.path.join(input_data["output_dir"],"res_1","saved_smis.json")))
    res_dict = {smi:float(pred) for smi, pred in zip(smis, pred_ensemble)}
    output_json = os.path.join(input_data["output_dir"], "result.json")
    json.dump(res_dict, open(output_json, "w"))

    return {"out_path": output_json}