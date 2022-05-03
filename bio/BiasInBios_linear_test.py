import torch
import wandb
from transformers import set_seed

from Data import BiasInBiosDataLinear
from Trainer import NoFinetuningBiasInBiosTrainer
sys.path.append('../src')
from ScriptUtils import parse_testing_args, log_test_results


def init_wandb(args):
    wandb.init(project="Bias in Bios Testing No Finetuning", config={
        "architecture": "RoBERTa with linear classifier",
        "finetuning": False,
        "dataset": "Bias in Bios",
        "seed": args.seed,
        "batch size": args.batch_size,
        "training data type": args.training_data,
        "testing data type": args.testing_data,
        "training balancing": args.training_balanced,
        "testing balancing": args.testing_balanced,
        "split type": args.split,
        "optimizer": "adamW"
    })


def __main__():
    args = parse_testing_args()
    seed = args.seed
    set_seed(seed)
    init_wandb(args)

    folder = "../data/biosbias"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = BiasInBiosDataLinear(folder + f"/vectors_{args.testing_data}_roberta-base_128.pt", seed, split=args.split,
                                      balanced=args.testing_balanced)

    batch_size = args.batch_size
    model = torch.nn.Linear(768, data.n_labels)
    loss_fn = torch.nn.CrossEntropyLoss()
    checkpoint_path = f"checkpoints/roberta-base/linear/{args.training_data}/{args.training_balanced}/seed_{seed}/best_model.pt"

    trainer = NoFinetuningBiasInBiosTrainer(model, loss_fn, None, batch_size, device=device)
    trainer.load_checkpoint(checkpoint_path)
    res = trainer.evaluate(data, args.split)

    log_test_results(res)

if __name__ == "__main__":
    __main__()