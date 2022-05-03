import wandb
import sys

from MDLProbingUtils import run_MDL_probing, general_MDL_args

sys.path.append('../src')
from ScriptUtils import load_winobias_vectors


def parse_args():
    parser = general_MDL_args()
    parser.add_argument('--model', required=True, type=str, help='the model type',
                        choices=["basic", "finetuned", "random"])
    parser.add_argument('--training_task', default="coref", required=False, type=str, help='the model original training task',
                        choices=["coref"])
    parser.add_argument('--training_balanced', type=str, help='balancing of the training data',
                        choices=["balanced", "imbalanced", "anon", "CA"], default=None)
    parser.add_argument('--training_data', type=str, help='data type of the training data, for biasbios task',
                        choices=["raw", "scrubbed"], default=None)
    parser.add_argument('--model_number', '-n', type=int, help='the model number to check on', required=False)

    args = parser.parse_args()
    print(args)
    return args


def init_wandb(args):
    wandb.init(project="Winobias MDL probing", config={
        "architecture": "RoBERTa",
        "seed": args.seed,
        "training balancing": args.training_balanced,
        "training_data": args.training_data,
        "model": args.model,
        "model_number": args.model_number,
        "model_seed": args.model_seed,
        "training_task": args.training_task,
    })


def main():
    args = parse_args()
    init_wandb(args)

    task_name = f'winobias_model_{args.training_task}_{args.model}_training_{args.training_balanced}_seed_{args.seed}_number_{args.model_number}'
    run_MDL_probing(args, load_winobias_vectors, task_name, shuffle=False)

if __name__ == '__main__':
    main()