import torch
import wandb
from transformers import AdamW, set_seed
from Data import BiasInBiosDataFinetuning
from Models import roBERTa_classifier, deBERTa_classifier
from Trainer import FinetuningBiasInBiosTrainer
sys.path.append('../src')
from ScriptUtils import parse_training_args


def get_project_name():
    name_of_project = "Bias in Bios"

    return name_of_project

def init_wandb(args):
    name_of_project = get_project_name()

    if args.model == 'roberta-base':
        model_name = 'RoBERTa'
    else:
        model_name = 'DeBERTa'

    wandb.init(project=name_of_project, config={
        "learning_rate": args.lr,
        "architecture": f"{model_name} with linear classifier",
        "finetuning": True,
        "dataset": "Bias in Bios",
        "seed": args.seed,
        "batch size": args.batch_size,
        "data type": args.data,
        "balancing": args.balanced,
        "total epochs": args.epochs,
        "checkpoint every": args.checkpointevery,
        "optimizer": "adamW",
    })

def get_data(args):
    data_train, data_valid = None, None
    data_path = f"../data/biosbias/tokens_{args.data}_{args.model}_128.pt"

    data_train = BiasInBiosDataFinetuning(data_path, args.seed, "train", args.balanced)
    data_valid = BiasInBiosDataFinetuning(data_path, args.seed, "valid", args.balanced)

    return data_train, data_valid

def get_model(args, n_labels):

    if args.model == 'roberta-base':
        model = roBERTa_classifier(n_labels)
    if args.model == 'deberta-base':
        model = deBERTa_classifier(n_labels)
    checkpoint_folder = f"checkpoints/{args.model}/finetuning/{args.data}/{args.balanced}/seed_{args.seed}"

    return model, checkpoint_folder

def get_trainer(args, model):
    loss_fn = torch.nn.CrossEntropyLoss()
    learning_rate = args.lr
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = FinetuningBiasInBiosTrainer(model, loss_fn, optimizer, args.batch_size, device=device)

    return trainer

def __main__():
    args = parse_training_args()
    init_wandb(args)

    set_seed(args.seed)
    data_train, data_valid = get_data(args)

    model, checkpoint_folder = get_model(args, data_train.n_labels)
    trainer = get_trainer(args, model)
    trainer.fit(data_train, data_valid, args.epochs, checkpoint_folder, checkpoint_every=args.checkpointevery, print_every=args.printevery)

if __name__ == "__main__":
    __main__()