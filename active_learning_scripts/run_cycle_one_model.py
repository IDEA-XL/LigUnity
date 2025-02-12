import pandas as pd
import numpy as np
import subprocess
import os
from pathlib import Path
import random
import argparse
import json


def parse_arguments():
    parser = argparse.ArgumentParser(description='Active Learning Cycle for Ligand Prediction')

    # Input/Output arguments
    parser.add_argument('--input_file', type=str, required=True,
                        help='Input CSV file containing ligand data (e.g., tyk2_fep.csv)')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Base directory for storing all results')
    parser.add_argument('--al_batch_size', type=int, required=True,
                        help='Number of samples for each active learning batch')

    # Experiment configuration
    parser.add_argument('--num_repeats', type=int, default=5,
                        help='Number of repeated experiments (default: 5)')
    parser.add_argument('--num_cycles', type=int, required=True,
                        help='Number of active learning cycles')

    # Model configuration
    parser.add_argument('--arch', type=str, required=True,
                        help='Model architecture')
    parser.add_argument('--weight_path', type=str, required=True,
                        help='Path to pretrained model weights')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--master_port', type=int, default=29500,
                        help='Master port for distributed training (default: 29500)')
    parser.add_argument('--device', type=int, default=0,
                        help='Device to run the model on (default: cuda:0)')
    parser.add_argument('--begin_greedy', type=int, default=0,
                        help='iter of begin to be pure greedy, using half greedy before')

    # Random seed
    parser.add_argument('--base_seed', type=int, default=42,
                        help='Base random seed (default: 42)')

    return parser.parse_args()


def run_model(arch, weight_path, results_path, result_file, lr, master_port, train_ligf, test_ligf, device):
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [
        "bash", "./active_learning_scripts/run_model.sh",
        arch,
        weight_path,
        results_path,
        result_file,
        str(lr),
        str(master_port),
        train_ligf,
        test_ligf,
        str(device)
    ]
    subprocess.run(cmd, check=True, cwd=project_root)


def prepare_initial_split(input_file, results_dir, al_batch_size, repeat_idx, cycle_idx, base_seed):
    # Read all ligands
    df = pd.read_csv(input_file)

    # Set random seed for reproducibility
    random.seed(base_seed + repeat_idx)  # Different seed for each repeat

    # Randomly select ligands for training and testing
    all_indices = list(range(len(df)))
    train_indices = random.sample(all_indices, al_batch_size)
    test_indices = [i for i in all_indices if i not in train_indices]

    # Create train and test files
    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]

    # Create file names with repeat and cycle information
    train_file = os.path.join(results_dir, f"repeat_{repeat_idx}_cycle_{cycle_idx}_train.csv")
    test_file = os.path.join(results_dir, f"repeat_{repeat_idx}_cycle_{cycle_idx}_test.csv")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(train_file), exist_ok=True)

    # Save files
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    return train_file, test_file


def read_jsonl_predictions(results_path, result_file):
    """
    Read predictions from jsonl file and calculate average predictions
    Returns a dictionary mapping SMILES to average predictions
    """
    predictions = {}
    all_predictions = []
    smiles_list = None

    jsonl_path = os.path.join(results_path, result_file)
    with open(jsonl_path, 'r') as f:
        # Read first line to get SMILES list
        first_line = f.readline()
        smiles_list = json.loads(first_line.strip())["tyk2"]["smiles"]

        # Read rest of lines containing predictions
        for line in f:
            pred_line = json.loads(line.strip())
            all_predictions.append(pred_line["tyk2"]["pred"])

    # Convert to numpy array for easier computation
    pred_array = np.array(all_predictions)
    # Calculate mean predictions
    mean_predictions = np.mean(pred_array, axis=0)

    # Create dictionary mapping SMILES to average predictions
    for smile, pred in zip(smiles_list, mean_predictions):
        predictions[smile] = float(pred)

    return predictions


def update_splits(results_dir, results_path, result_file, prev_train_file, prev_test_file, repeat_idx, cycle_idx,
                  al_batch_size, begin_greedy):
    # Read predictions from jsonl file
    predictions = read_jsonl_predictions(results_path, result_file)

    # Read previous test file
    test_df = pd.read_csv(prev_test_file)

    # Add predictions to test_df
    test_df['prediction'] = test_df['Smiles'].map(predictions)

    # Sort by predictions (high to low)
    test_df_sorted = test_df.sort_values('prediction', ascending=False)

    # Read previous train file
    train_df = pd.read_csv(prev_train_file)

    # Create new file names
    new_train_file = os.path.join(results_dir, f"repeat_{repeat_idx}_cycle_{cycle_idx}_train.csv")
    new_test_file = os.path.join(results_dir, f"repeat_{repeat_idx}_cycle_{cycle_idx}_test.csv")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(new_train_file), exist_ok=True)

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
    combined_train_df = pd.concat([train_df, new_train_compounds])

    for _ in range(3):
        print("########################################")
    print("Cycling: ", cycle_idx)
    print("top_1p: {}/100".format(combined_train_df['top_1p'].sum()))
    print("top_2p: {}/200".format(combined_train_df['top_2p'].sum()))
    print("top_5p: {}/500".format(combined_train_df['top_5p'].sum()))

    # Save files
    combined_train_df.to_csv(new_train_file, index=False)
    remaining_test_compounds.to_csv(new_test_file, index=False)

    return new_train_file, new_test_file


def run_active_learning(args):
    # Create base results directory
    os.system(f"rm -rf {args.results_dir}")
    os.makedirs(args.results_dir, exist_ok=True)

    for repeat_idx in range(args.num_repeats):
        print(f"Starting repeat {repeat_idx}")

        # Initial split for this repeat
        train_file, test_file = prepare_initial_split(
            args.input_file,
            args.results_dir,
            args.al_batch_size,
            repeat_idx,
            0,  # First cycle
            args.base_seed
        )

        for cycle_idx in range(args.num_cycles):
            print(f"Running cycle {cycle_idx} for repeat {repeat_idx}")

            # Create results directory for this cycle
            results_path = args.results_dir

            # Result file name
            result_file = f"repeat_{repeat_idx}_cycle_{cycle_idx}_results.jsonl"
            if os.path.exists(f"{args.results_dir}/{result_file}"):
                os.remove(f"{args.results_dir}/{result_file}")

            # Run the model
            run_model(
                arch=args.arch,
                weight_path=args.weight_path,
                results_path=results_path,
                result_file=result_file,
                lr=args.lr,
                master_port=args.master_port,
                train_ligf=train_file,
                test_ligf=test_file,
                device=args.device
            )

            # Update splits for next cycle
            if cycle_idx < args.num_cycles - 1:  # Don't update after last cycle
                train_file, test_file = update_splits(
                    args.results_dir,
                    results_path,
                    result_file,
                    train_file,
                    test_file,
                    repeat_idx,
                    cycle_idx + 1,
                    args.al_batch_size,
                    args.begin_greedy
                )


if __name__ == "__main__":
    args = parse_arguments()
    run_active_learning(args)