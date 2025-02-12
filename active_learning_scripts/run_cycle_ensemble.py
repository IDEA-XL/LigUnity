import pandas as pd
import numpy as np
import subprocess
import os
from pathlib import Path
import random
import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, wait

def parse_arguments():
    parser = argparse.ArgumentParser(description='Active Learning Cycle for Ligand Prediction')

    # Input/Output arguments
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input CSV file containing ligand data (e.g., tyk2_fep.csv)')
    parser.add_argument('--results_dir_1', type=str, required=True,
                        help='Results directory for first model')
    parser.add_argument('--results_dir_2', type=str, required=True,
                        help='Results directory for second model')
    parser.add_argument('--al_batch_size', type=int, required=True,
                        help='Number of samples for each active learning batch')

    # Experiment configuration
    parser.add_argument('--num_repeats', type=int, default=5,
                        help='Number of repeated experiments (default: 5)')
    parser.add_argument('--num_cycles', type=int, required=True,
                        help='Number of active learning cycles')

    # Model configuration
    parser.add_argument('--arch_1', type=str, required=True,
                        help='First model architecture')
    parser.add_argument('--arch_2', type=str, required=True,
                        help='Second model architecture')
    parser.add_argument('--weight_path_1', type=str, required=True,
                        help='Path to first model pretrained weights')
    parser.add_argument('--weight_path_2', type=str, required=True,
                        help='Path to second model pretrained weights')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--master_port', type=int, default=29500,
                        help='Master port for distributed training (default: 29500)')
    parser.add_argument('--device', type=int, default=0,
                        help='Base device to run the models on (default: 0)')
    parser.add_argument('--begin_greedy', type=int, default=0,
                        help='iter of begin to be pure greedy, using half greedy before')

    # Random seed
    parser.add_argument('--base_seed', type=int, default=42,
                        help='Base random seed (default: 42)')

    return parser.parse_args()


def _run(cmd):
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run(cmd, check=True, cwd=project_root)


def run_model(arch_1, arch_2, weight_path_1, weight_path_2, results_path_1, results_path_2, result_file, lr,
              master_port, train_ligf, test_ligf, device):
    cmd1 = [
        "bash", "./active_learning_scripts/run_model.sh",
        arch_1,
        weight_path_1,
        results_path_1,
        result_file,
        str(lr),
        str(master_port),
        train_ligf,
        test_ligf,
        str(device)
    ]

    cmd2 = [
        "bash", "./active_learning_scripts/run_model.sh",
        arch_2,
        weight_path_2,
        results_path_2,
        result_file,
        str(lr),
        str(master_port + 1),
        train_ligf,
        test_ligf,
        str(device + 1)
    ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        task1 = executor.submit(_run, cmd=cmd1)
        task2 = executor.submit(_run, cmd=cmd2)
        wait([task1, task2])


def read_predictions(results_path, result_file):
    """
    Read predictions from a single model
    """
    predictions = {}

    jsonl_path = os.path.join(results_path, result_file)
    with open(jsonl_path, 'r') as f:
        first_line = json.loads(f.readline().strip())
        smiles_list = first_line["tyk2"]["smiles"]
        all_predictions = []
        for line in f:
            pred_line = json.loads(line.strip())
            all_predictions.append(pred_line["tyk2"]["pred"])

    # Convert to numpy array and calculate mean predictions
    pred_array = np.array(all_predictions)
    mean_predictions = np.mean(pred_array, axis=0)

    # Create dictionary mapping SMILES to predictions
    for smile, pred in zip(smiles_list, mean_predictions):
        predictions[smile] = float(pred)

    return predictions

def prepare_initial_split(input_file, results_dir_1, results_dir_2, al_batch_size, repeat_idx, cycle_idx, base_seed):
    # Read all ligands
    df = pd.read_csv(input_file)

    # Set random seed for reproducibility
    random.seed(base_seed + repeat_idx)

    # Randomly select ligands for training and testing
    all_indices = list(range(len(df)))
    train_indices = random.sample(all_indices, al_batch_size)
    test_indices = [i for i in all_indices if i not in train_indices]

    # Create train and test files
    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]

    # Create file names for both directories
    train_file_1 = os.path.join(results_dir_1, f"repeat_{repeat_idx}_cycle_{cycle_idx}_train.csv")
    test_file_1 = os.path.join(results_dir_1, f"repeat_{repeat_idx}_cycle_{cycle_idx}_test.csv")

    train_file_2 = os.path.join(results_dir_2, f"repeat_{repeat_idx}_cycle_{cycle_idx}_train.csv")
    test_file_2 = os.path.join(results_dir_2, f"repeat_{repeat_idx}_cycle_{cycle_idx}_test.csv")

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(train_file_1), exist_ok=True)
    os.makedirs(os.path.dirname(train_file_2), exist_ok=True)

    # Save files to both directories
    train_df.to_csv(train_file_1, index=False)
    test_df.to_csv(test_file_1, index=False)
    train_df.to_csv(train_file_2, index=False)
    test_df.to_csv(test_file_2, index=False)

    return train_file_1, test_file_1, train_file_2, test_file_2


def read_and_combine_predictions(results_path_1, results_path_2, result_file):
    """
    Read predictions from both models and calculate average predictions
    """
    predictions = {}

    # Read predictions from model 1
    jsonl_path_1 = os.path.join(results_path_1, result_file)
    with open(jsonl_path_1, 'r') as f:
        first_line = json.loads(f.readline().strip())
        smiles_list = first_line["tyk2"]["smiles"]
        all_predictions_1 = []
        for line in f:
            pred_line = json.loads(line.strip())
            all_predictions_1.append(pred_line["tyk2"]["pred"])

    # Read predictions from model 2
    jsonl_path_2 = os.path.join(results_path_2, result_file)
    with open(jsonl_path_2, 'r') as f:
        f.readline()  # skip first line as we already have smiles_list
        all_predictions_2 = []
        for line in f:
            pred_line = json.loads(line.strip())
            all_predictions_2.append(pred_line["tyk2"]["pred"])

    # Convert to numpy arrays
    pred_array_1 = np.array(all_predictions_1)
    pred_array_2 = np.array(all_predictions_2)

    # Calculate mean predictions across both models
    mean_predictions = (np.mean(pred_array_1, axis=0) + np.mean(pred_array_2, axis=0)) / 2

    # Create dictionary mapping SMILES to average predictions
    for smile, pred in zip(smiles_list, mean_predictions):
        predictions[smile] = float(pred)

    return predictions


def update_splits(results_dir_1, results_dir_2, predictions_1, predictions_2,
                 prev_train_file_1, prev_test_file_1,
                 prev_train_file_2, prev_test_file_2,
                 repeat_idx, cycle_idx, al_batch_size, begin_greedy):
    # Read previous test files
    test_df_1 = pd.read_csv(prev_test_file_1)
    test_df_2 = pd.read_csv(prev_test_file_2)

    # Add predictions to test_df
    test_df_1['prediction_1'] = test_df_1['Smiles'].map(predictions_1)
    test_df_1['prediction_2'] = test_df_1['Smiles'].map(predictions_2)
    test_df_1['prediction'] = (test_df_1['prediction_1'] + test_df_1['prediction_2']) / 2

    # Sort by average predictions (high to low)
    test_df_sorted = test_df_1.sort_values('prediction', ascending=False)

    # Read previous train files
    train_df_1 = pd.read_csv(prev_train_file_1)
    train_df_2 = pd.read_csv(prev_train_file_2)

    # Create new file names for both directories
    new_train_file_1 = os.path.join(results_dir_1, f"repeat_{repeat_idx}_cycle_{cycle_idx}_train.csv")
    new_test_file_1 = os.path.join(results_dir_1, f"repeat_{repeat_idx}_cycle_{cycle_idx}_test.csv")
    new_train_file_2 = os.path.join(results_dir_2, f"repeat_{repeat_idx}_cycle_{cycle_idx}_train.csv")
    new_test_file_2 = os.path.join(results_dir_2, f"repeat_{repeat_idx}_cycle_{cycle_idx}_test.csv")

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(new_train_file_1), exist_ok=True)
    os.makedirs(os.path.dirname(new_train_file_2), exist_ok=True)

    if cycle_idx >= begin_greedy:
        # Take top al_batch_size compounds for training
        new_train_compounds = test_df_sorted.head(al_batch_size)
        remaining_test_compounds = test_df_sorted.iloc[al_batch_size:]
    else:
        # use half greedy approach
        new_train_compounds_tmp_1 = test_df_sorted.head(al_batch_size//2)
        remaining_test_compounds_tmp = test_df_sorted.iloc[al_batch_size//2:]
        all_indices = list(range(len(remaining_test_compounds_tmp)))

        train_indices = random.sample(all_indices, al_batch_size - al_batch_size//2)
        test_indices = [i for i in all_indices if i not in train_indices]
        remaining_test_compounds = remaining_test_compounds_tmp.iloc[test_indices]
        new_train_compounds_tmp_2 = remaining_test_compounds_tmp.iloc[train_indices]
        new_train_compounds = pd.concat([new_train_compounds_tmp_1, new_train_compounds_tmp_2])

    # Combine with previous training data
    combined_train_df = pd.concat([train_df_1, new_train_compounds])

    for _ in range(3):
        print("########################################")
    print("Cycling: ", cycle_idx)
    print("top_1p: {}/100".format(combined_train_df['top_1p'].sum()))
    print("top_2p: {}/200".format(combined_train_df['top_2p'].sum()))
    print("top_5p: {}/500".format(combined_train_df['top_5p'].sum()))

    # Save files for both models (same content, different directories)
    combined_train_df.to_csv(new_train_file_1, index=False)
    remaining_test_compounds.to_csv(new_test_file_1, index=False)
    combined_train_df.to_csv(new_train_file_2, index=False)
    remaining_test_compounds.to_csv(new_test_file_2, index=False)

    return (new_train_file_1, new_test_file_1,
            new_train_file_2, new_test_file_2)


def run_active_learning(args):
    # Create base results directories
    os.system(f"rm -rf {args.results_dir_1}")
    os.system(f"rm -rf {args.results_dir_2}")
    os.makedirs(args.results_dir_1, exist_ok=True)
    os.makedirs(args.results_dir_2, exist_ok=True)

    for repeat_idx in range(args.num_repeats):
        print(f"Starting repeat {repeat_idx}")

        # Initial split for this repeat
        train_file_1, test_file_1, train_file_2, test_file_2 = prepare_initial_split(
            args.input_file,
            args.results_dir_1,
            args.results_dir_2,
            args.al_batch_size,
            repeat_idx,
            0,  # First cycle
            args.base_seed
        )

        for cycle_idx in range(args.num_cycles):
            print(f"Running cycle {cycle_idx} for repeat {repeat_idx}")

            # Result file name
            result_file = f"repeat_{repeat_idx}_cycle_{cycle_idx}_results.jsonl"
            if os.path.exists(f"{args.results_dir_1}/{result_file}"):
                os.remove(f"{args.results_dir_1}/{result_file}")
            if os.path.exists(f"{args.results_dir_2}/{result_file}"):
                os.remove(f"{args.results_dir_2}/{result_file}")

            # Run both models
            run_model(
                arch_1=args.arch_1,
                arch_2=args.arch_2,
                weight_path_1=args.weight_path_1,
                weight_path_2=args.weight_path_2,
                results_path_1=args.results_dir_1,
                results_path_2=args.results_dir_2,
                result_file=result_file,
                lr=args.lr,
                master_port=args.master_port,
                train_ligf=train_file_1,
                test_ligf=test_file_1,
                device=args.device
            )

            # Update splits for next cycle
            if cycle_idx < args.num_cycles - 1:
                # Read predictions from both models separately
                predictions_1 = read_predictions(args.results_dir_1, result_file)
                predictions_2 = read_predictions(args.results_dir_2, result_file)

                # Update splits for both models
                train_file_1, test_file_1, train_file_2, test_file_2 = update_splits(
                    args.results_dir_1,
                    args.results_dir_2,
                    predictions_1,
                    predictions_2,
                    train_file_1,
                    test_file_1,
                    train_file_2,
                    test_file_2,
                    repeat_idx,
                    cycle_idx + 1,
                    args.al_batch_size,
                    args.begin_greedy
                )


if __name__ == "__main__":
    args = parse_arguments()
    run_active_learning(args)