import argparse
import random
import torch
import numpy as np
import wandb
from transformers import AdamW, set_seed

from Data import BiasInBiosDataLinear
from ScriptUtils import parse_training_args, log_test_results, parse_testing_args
from Trainer import NoFinetuningBiasInBiosTrainer


# def parse_args():
#     parser = argparse.ArgumentParser(description='Run testing on Bias in Bios dataset with top layer replacement.')
#
#     parser.add_argument('--batch_size', '-b', default=64, type=int, help='the batch size to test with')
#     parser.add_argument('--training_data', required=False, type=str, help='the data type the top layer was trained on',
#                         choices=["raw", "scrubbed", "name"], default='raw')
#     parser.add_argument('--testing_data_type', required=False, type=str, help='the data type the model will be tested on',
#                         choices=["raw", "scrubbed", "name"], default='raw')
#     parser.add_argument('--training_balanced', type=str, help='balancing of the data the top layer was trained on',
#                         choices=["subsampled", "oversampled", "original"], default="original")
#     parser.add_argument('--testing_balanced', type=str, help='balancing of the data the model will be tested',
#                         choices=["subsampled", "oversampled", "original"], default="original")
#     parser.add_argument('--split', type=str, help='the split type to test',
#                         choices=["train", "test", "valid"])
#     parser.add_argument('--seed', '-s', default=0, type=int, help='the random seed')
#     parser.add_argument('--embedding_training_data', required=False, type=str,
#                         help='the data type of pretrained embeddings',
#                         choices=["raw", "scrubbed", "name"], default='raw')
#     parser.add_argument('--embedding_training_balanced', type=str,
#                         help='balancing of the data for pretrained embeddings',
#                         choices=["subsampled", "oversampled", "original"], default="original")
#     parser.add_argument('--embedding_seed', default=None, type=int, help='the random seed embedding was trained with')
#
#     args = parser.parse_args()
#     print(args)
#
#     return args

def init_wandb(args):
    if args.embedding_seed is not None:
        project_name = "Bias in Bios replace top layer separate seeds test"
        # project_name = "Bias in Bios replace top layer separate seeds test with US occupation statistics"
    else:
        project_name = "Bias in Bios replace top layer test"

    if args.model == 'roberta-base':
        model_name = 'RoBERTa'
    else:
        model_name = 'DeBERTa'

    wandb.init(project=project_name, config={
        "architecture": f"{model_name} with linear classifier",
        "finetuning": False,
        "seed": args.seed,
        "embedding_seed": args.embedding_seed,
        "batch size": args.batch_size,
        "training_balanced": args.training_balanced,
        "testing_balanced": args.testing_balanced,
        "training_data": args.training_data,
        "testing_data": args.testing_data,
        "embedding_training_data": args.embedding_training_data,
        "embedding_training_balanced": args.embedding_training_balanced,
        "split": args.split
    })

def __main__():
    args = parse_testing_args(replace_top_layer=True)
    seed = args.seed
    set_seed(seed)
    init_wandb(args)

    folder = f"../data/biosbias/vectors_extracted_from_trained_models/{args.model}/finetuning/" \
             f"{args.embedding_training_data}/{args.embedding_training_balanced}/seed_{args.embedding_seed}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = BiasInBiosDataLinear(folder + f"/vectors_{args.testing_data}_{args.model}_128.pt", args.embedding_seed, split=args.split,
                                      balanced=args.testing_balanced)

    # Now change to random seed for training
    # set_seed(seed)

    batch_size = args.batch_size
    model = torch.nn.Linear(768, data.n_labels)
    loss_fn = torch.nn.CrossEntropyLoss()

    if args.embedding_seed is not None:
        checkpoint_path = f"checkpoints/bias_in_bios/{args.model}/linear/replace_top_layer/{args.embedding_training_data}/{args.embedding_training_balanced}/{args.training_data}/{args.training_balanced}/seed_{seed}/embedding_seed_{args.embedding_seed}/best_model.pt"
    else:
        checkpoint_path = f"checkpoints/bias_in_bios/{args.model}/linear/replace_top_layer/{args.embedding_training_data}/{args.embedding_training_balanced}/{args.training_data}/{args.training_balanced}/seed_{seed}/best_model.pt"

    trainer = NoFinetuningBiasInBiosTrainer(model, loss_fn, None, batch_size, device=device)
    trainer.load_checkpoint(checkpoint_path)
    res = trainer.evaluate(data, args.split)

    log_test_results(res)

if __name__ == "__main__":
    __main__()