import random
import torch
import numpy as np
import wandb
from transformers import AdamW

from Data import BiasInBiosDataLinear
from ScriptUtils import parse_training_args
from Trainer import NoFinetuningBiasInBiosTrainer

args = parse_training_args(replace_top_layer=True)
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = args.seed
if args.embedding_seed is not None:
    embedding_seed = args.embedding_seed
    project_name = "Bias in Bios replace top layer separate seeds"
else:
    embedding_seed = seed
    project_name = "Bias in Bios replace top layer"

if args.model == 'roberta-base':
    model_name = 'RoBERTa'
else:
    model_name = 'DeBERTa'
wandb.init(project=project_name, config={
    "learning_rate": args.lr,
    "architecture": f"{model_name} with linear classifier",
    "finetuning": False,
    "seed": seed,
    "embedding_seed": args.embedding_seed,
    "batch_size": args.batch_size,
    "data_type": args.data,
    "balancing": args.balanced,
    "total epochs": args.epochs,
    "optimizer": "adamW",
    "embedding_training_data": args.embedding_training_data,
    "embedding_training_balanced": args.embedding_training_balanced
})

batch_size = args.batch_size
folder = f"../data/biosbias/vectors_extracted_from_trained_models/{args.model}/finetuning/" \
         f"{args.embedding_training_data}/{args.embedding_training_balanced}/seed_{embedding_seed}"

data_train = BiasInBiosDataLinear(folder + f"/vectors_{args.data}_{args.model}_128.pt", embedding_seed, split="train", balanced=args.balanced)
data_valid = BiasInBiosDataLinear(folder + f"/vectors_{args.data}_{args.model}_128.pt", embedding_seed, split="valid", balanced=args.balanced)

model = torch.nn.Linear(768, data_train.n_labels)
learning_rate = args.lr
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_epochs = args.epochs
checkpoint_folder = f"checkpoints/bias_in_bios/{args.model}/linear/replace_top_layer/{args.embedding_training_data}/{args.embedding_training_balanced}/{args.data}/{args.balanced}/seed_{seed}/embedding_seed_{embedding_seed}"

trainer = NoFinetuningBiasInBiosTrainer(model, loss_fn, optimizer, batch_size, device=device)
trainer.fit(data_train, data_valid, num_epochs, checkpoint_folder, print_every=args.printevery, checkpoint_every=args.checkpointevery)
