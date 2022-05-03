import torch

from run import Runner
import sys


def evaluate(config_name, gpu_id, saved_suffix):
    runner = Runner(config_name, gpu_id)
    model = runner.initialize_model(saved_suffix)

    _, _, examples_test = runner.data.get_tensor_examples(train=False)
    stored_info = runner.data.get_stored_info()

    predictions = runner.get_predictions(model, examples_test, stored_info)  # Eval dev
    torch.save(predictions, f"preds/pred_{saved_suffix}_{config_name}.pt")
    # predictions = runner.get_predictions(model, examples_dev, stored_info)  # Eval dev
    # torch.save(predictions, f"pred_{saved_suffix}_{config_name}_dev.pt")


if __name__ == '__main__':
    config_name, saved_suffix, gpu_id = sys.argv[1], sys.argv[2], int(sys.argv[3])
    evaluate(config_name, gpu_id, saved_suffix)
