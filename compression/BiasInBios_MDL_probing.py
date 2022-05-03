
import sys
import wandb
from MDLProbingUtils import run_MDL_probing, general_MDL_args
sys.path.append('../src')
from ScriptUtils import load_bias_in_bios_vectors

def parse_args():
    parser = general_MDL_args()
    parser.add_argument('--model', required=True, type=str, help='the model type',
                        choices=["basic", "finetuning", "random"])
    parser.add_argument('--feature_extractor', default='roberta-base', type=str, choices=['roberta-base', 'deberta-base'],
                        help='model to use as a feature extractor')
    parser.add_argument('--training_data', required=False, type=str, help='the data type the model was trained on',
                        choices=["raw", "scrubbed"], default=None)
    parser.add_argument('--training_balanced', type=str, help='balancing of the training data',
                        choices=["subsampled", "oversampled", "original"], default="original")
    parser.add_argument('--type', type=str, help='the type of vectors to probe',
                        choices=["raw", "scrubbed"])
    parser.add_argument('--testing_balanced', type=str, help='balancing of the testing data',
                        choices=["subsampled", "oversampled", "original"], default="original")


    args = parser.parse_args()
    print(args)
    return args


def init_wandb(args):
    if args.feature_extractor == 'roberta-base':
        architecture = "RoBERTa"
    else:
        architecture = "DeBERTa"

    wandb.init(project="Bias in Bios MDL probing", config={
        "architecture": architecture,
        "seed": args.seed,
        "model_seed": args.model_seed,
        "training data": args.training_data,
        "training balancing": args.training_balanced,
        "testing balancing": args.testing_balanced,
        "type": args.type,
        "model": args.model,
    })


def main():
    args = parse_args()
    init_wandb(args)

    task_name = f'biasinbios_model_{args.model}_type_{args.type}_{args.testing_balanced}_training_{args.training_data}_{args.training_balanced}_seed_{args.seed}'
    run_MDL_probing(args, load_bias_in_bios_vectors, task_name, shuffle=True)

if __name__ == '__main__':
    main()