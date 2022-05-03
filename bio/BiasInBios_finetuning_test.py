import torch
import wandb
from transformers import set_seed

from Data import BiasInBiosDataFinetuning
from Models import roBERTa_classifier, deBERTa_classifier
sys.path.append('../src')
from ScriptUtils import parse_testing_args, log_test_results
from Trainer import FinetuningBiasInBiosTrainer


def get_project_name():
    name_of_project = "Bias in Bios Testing"

    return name_of_project

def init_wandb(args):
    name_of_project = get_project_name()

    if args.model == 'roberta-base':
        model_name = 'RoBERTa'
    else:
        model_name = 'DeBERTa'


    wandb.init(project=name_of_project, config={
        "architecture": f"{model_name} with linear classifier",
        "finetuning": True,
        "dataset": "Bias in Bios",
        "seed": args.seed,
        "batch size": args.batch_size,
        "training data type": args.training_data,
        "testing data type": args.testing_data,
        "training balancing": args.training_balanced,
        "testing balancing": args.testing_balanced,
        "split type": args.split,
        "optimizer": "adamW",
    })

def __main__():
    args = parse_testing_args()
    seed = args.seed
    set_seed(seed)
    init_wandb(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    data = BiasInBiosDataFinetuning(f"../data/biosbias/tokens_{args.testing_data}_{args.model}_128.pt",
                                          args.seed, args.split, args.testing_balanced)

    if args.model == 'roberta-base':
        model = roBERTa_classifier(data.n_labels)
    if args.model == 'deberta-base':
        model = deBERTa_classifier(data.n_labels)

    checkpoint_path = f"checkpoints/{args.model}/finetuning/{args.training_data}/{args.training_balanced}/seed_{args.seed}/best_model.pt"

    loss_fn = torch.nn.CrossEntropyLoss()
    trainer = FinetuningBiasInBiosTrainer(model, loss_fn, None, batch_size, device=device)
    trainer.load_checkpoint(checkpoint_path)

    res = trainer.evaluate(data, args.split)
    log_test_results(res)

if __name__ == "__main__":
    __main__()