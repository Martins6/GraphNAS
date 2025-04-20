"""Entry point."""
from loguru import logger
import argparse
from datetime import datetime
import time
import json

import torch

import graphnas.trainer as trainer
import graphnas.utils.tensor_utils as utils


def build_args():
    parser = argparse.ArgumentParser(description='GraphNAS')
    register_default_args(parser)
    args = parser.parse_args()

    return args


def register_default_args(parser):
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'derive'],
                        help='train: Training GraphNAS, derive: Deriving Architectures')
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument("--cuda", type=bool, default=True, required=False,
                        help="run in cuda mode")
    parser.add_argument('--save_epoch', type=int, default=2)
    parser.add_argument('--max_save_num', type=int, default=5)
    # controller
    parser.add_argument('--layers_of_child_model', type=int, default=2)
    parser.add_argument('--shared_initial_step', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--entropy_mode', type=str, default='reward', choices=['reward', 'regularizer'])
    parser.add_argument('--entropy_coeff', type=float, default=1e-4)
    parser.add_argument('--shared_rnn_max_length', type=int, default=35)
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--search_mode', type=str, default='macro')
    parser.add_argument('--format', type=str, default='two')
    parser.add_argument('--max_epoch', type=int, default=10)

    parser.add_argument('--ema_baseline_decay', type=float, default=0.95)
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--controller_max_step', type=int, default=100,
                        help='step for controller parameters')
    parser.add_argument('--controller_optim', type=str, default='adam')
    parser.add_argument('--controller_lr', type=float, default=3.5e-4,
                        help="will be ignored if --controller_lr_cosine=True")
    parser.add_argument('--controller_grad_clip', type=float, default=0)
    parser.add_argument('--tanh_c', type=float, default=2.5)
    parser.add_argument('--softmax_temperature', type=float, default=5.0)
    parser.add_argument('--derive_num_sample', type=int, default=100)
    parser.add_argument('--derive_finally', type=bool, default=True)
    parser.add_argument('--derive_from_history', type=bool, default=True)

    # child model
    parser.add_argument("--dataset", type=str, default="Cora", required=False,
                        help="The input dataset.")
    parser.add_argument("--epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--retrain_epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--multi_label", type=bool, default=False,
                        help="multi_label or single_label task")
    parser.add_argument("--residual", action="store_false",
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0.6,
                        help="input feature dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--param_file", type=str, default="cora_test.pkl",
                        help="learning rate")
    parser.add_argument("--optim_file", type=str, default="opt_cora_test.pkl",
                        help="optimizer save path")
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--max_param', type=float, default=5E6)
    parser.add_argument('--supervised', type=bool, default=False)
    parser.add_argument('--submanager_log_file', type=str, default=f"sub_manager_logger_file_{time.time()}.txt")


def main(args):  # pylint:disable=redefined-outer-name

    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    utils.makedirs(args.dataset)

    trnr = trainer.Trainer(args)

    if args.mode == 'train':
        print(args)
        best_actions, best_score = trnr.train()
    elif args.mode == 'derive':
        best_actions, best_score = trnr.derive()
    else:
        raise Exception(f"[!] Mode not found: {args.mode}")
    
    return best_actions, best_score



def log_experiments_diff_seed(args, seeds, experiment_settings):
    logger.info(experiment_settings)
    logger.info(args)

    # Create initial results dictionary
    results = {
        "experiment_settings": experiment_settings,
        "args": vars(args),  # Convert args namespace to dict
        "runs": []
    }

    # Create JSON file with initial structure
    date_time = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    json_filename = f"adriel_experiment_results_{date_time}.json"
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Created results file: {json_filename}")

    for seed in seeds:
        start_time = datetime.now()
        args.random_seed = seed
        best_actions, best_score = main(args)
        end_time = datetime.now()
        
        # Calculate duration in seconds
        duration_seconds = (end_time - start_time).total_seconds()
        
        # Log to console
        logger.info(f"seed: {seed}, best actions: {best_actions}, best score: {best_score}, duration: {duration_seconds:.2f} seconds")
        
        # Create run data
        run_data = {
            "seed": seed,
            "best_actions": best_actions,
            "best_score": best_score,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration_seconds
        }

        # Read existing results
        with open(json_filename, 'r') as f:
            results = json.load(f)
        
        # Append new run data
        results["runs"].append(run_data)
        
        # Write updated results back to file
        with open(json_filename, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Updated results file with seed {seed}")




if __name__ == "__main__":
    args = build_args()

    for dict_data_transf in [
        # {"dataset": "Cora", "normalize_features": True, "sample": True},
        # {"dataset": "CiteSeer", "normalize_features": True, "sample": True},
        # {"dataset": "PubMed", "normalize_features": True, "sample": True},
        # {"dataset": "Photo", "normalize_features": True, "sample": True},
        # {"dataset": "Computers", "normalize_features": True, "sample": True},
        {"dataset": "Cornell", "normalize_features": True, "sample": True},
        {"dataset": "Texas", "normalize_features": True, "sample": True},
        {"dataset": "Wisconsin", "normalize_features": True, "sample": True},
    ]:
        args.normalize_features = dict_data_transf["normalize_features"]
        args.random_node_split = dict_data_transf["sample"]
        args.dataset= dict_data_transf["dataset"]

        if args.dataset in ["Cora", "CiteSeer"]:
            args.entropy_coeff = 1e-4
            args.lr = 0.005
        if args.dataset == "PubMed":
            args.entropy_coeff = 1e-3
            args.lr = 0.01
        

        try:
            date_time = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
            logger_id = logger.add(
                f".logs/graphnas_experiment_{date_time}.log",
                format="{time:YYYY-MM-DDTHH:MM:SS} | {level} | {message}",
                level="INFO",
            )
            log_experiments_diff_seed(
                args,
                [123, 42, 1], # [123, 42, 1, 345678910, 7],
                dict_data_transf,
            )
            logger.info("Experiment completed successfully.")
        except Exception as e:
            print(e)
            logger.exception(f"Error in experiment: {e}")
            # log traceback
        logger.remove(logger_id)
    
    # main(args)
