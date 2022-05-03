import random
import numpy as np
import torch
import wandb
from transformers import AdamW

from Data import BiasInBiosDataLinear
from Trainer import NoFinetuningBiasInBiosTrainer
sys.path.append('../src')
from ScriptUtils import parse_training_args

args = parse_training_args()
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(project="Bias in Bios linear training", config={
    "learning_rate": args.lr,
    "architecture": "RoBERTa with linear classifier",
    "finetuning": False,
    "dataset": "Bias in Bios",
    "seed": seed,
    "batch size": args.batch_size,
    "data type": args.data,
    "balancing": args.balanced,
    "total epochs": args.epochs,
    "optimizer": "adamW"
})

batch_size = args.batch_size

folder = "../data/biosbias"

data_train = BiasInBiosDataLinear(folder + f"/vectors_{args.data}_roberta-base_128.pt", seed, split="train", balanced=args.balanced)
data_valid = BiasInBiosDataLinear(folder + f"/vectors_{args.data}_roberta-base_128.pt", seed, split="valid", balanced=args.balanced)

model = torch.nn.Linear(768, data_train.n_labels)
learning_rate = args.lr
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_epochs = args.epochs
checkpoint_folder = f"checkpoints/roberta-base/linear/{args.data}/{args.balanced}/seed_{seed}"

trainer = NoFinetuningBiasInBiosTrainer(model, loss_fn, optimizer, batch_size, device=device)
trainer.fit(data_train, data_valid, num_epochs, checkpoint_folder, print_every=args.printevery, checkpoint_every=args.checkpointevery)
