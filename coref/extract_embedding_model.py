import argparse
from pathlib import Path

import torch

from run import Runner

parser = argparse.ArgumentParser(description='Extract the embedding model from the coref model.')
parser.add_argument('--training_balanced', type=str, help='balancing of training data used to train the model',
                    choices=["balanced", "imbalanced", "CA", "anon"], default=None)
parser.add_argument('--numbers', '-n', nargs="+", default=0, type=int, help='the model numbers (each model is trained with a different seed)')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_name = "test_roberta_base" if args.training_balanced == "imbalanced" else "test_roberta_base_balanced_model"

runner = Runner(config_name, 0)
for num in args.numbers:
    saved_suffix = args.training_balanced + str(num)
    model = runner.initialize_model(saved_suffix).bert
    folder_ckpt = f"embedding_models/roberta-base/{args.training_balanced}/{num}"
    Path(folder_ckpt).mkdir(parents=True, exist_ok=True)
    path_ckpt = folder_ckpt + "/model.pt"
    torch.save(model.state_dict(), path_ckpt)