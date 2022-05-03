import argparse
import random

import torch
from transformers import RobertaModel, RobertaForMaskedLM, RobertaConfig, AutoModel, DebertaTokenizer, DebertaModel, \
    AutoTokenizer, DebertaConfig
from transformers import RobertaTokenizer
import numpy as np
from DataUtils import BiasInBios_extract_vectors_data
from Models import roBERTa_classifier, deBERTa_classifier
from Trainer import load_checkpoint

N_LABELS = 28

def parse_args():
    parser = argparse.ArgumentParser(description='Extracting vectors from trained Bias in Bios model.')
    parser.add_argument('--model', required=True, type=str, help='the model type',
                        choices=["basic", "finetuning", "random"])
    parser.add_argument('--feature_extractor', default='roberta-base', type=str, choices=['roberta-base', 'deberta-base'],
                        help='model to use as a feature extractor')
    parser.add_argument('--training_data', type=str, help='type of training data used to train the model',
                        choices=["raw", "scrubbed"], default=None)
    parser.add_argument('--training_balanced', type=str, help='balancing of training data used to train the model',
                        choices=["subsampled", "oversampled", "original"], default="original")
    parser.add_argument('--seed', '-s', default=0, type=int, help='the random seed the model was trained on')
    parser.add_argument('--data', '-d', type=str, help='type of data to extract',
                        choices=["raw", "scrubbed", "name", "scrubbed_extra"], default='raw')

    args = parser.parse_args()
    print("model:", args.model)
    print("feature_extractor:", args.feature_extractor)
    print("trained on:", args.training_data, args.training_balanced)
    print("seed:", args.seed)

    return args

def get_model(args):
    global model, folder
    if args.model == "basic":
        if args.feature_extractor == 'roberta-base':
            model = RobertaModel.from_pretrained('roberta-base', output_attentions=True, output_hidden_states=True)
        else:
            model = DebertaModel.from_pretrained('microsoft/deberta-base', output_attentions=False,
                                                 output_hidden_states=False)

        folder = f"../data/biosbias/vectors_extracted_from_trained_models/{args.feature_extractor}/basic/seed_{args.seed}"
    if args.model == "random":
        if args.feature_extractor == 'roberta-base':
            configuration = RobertaConfig()
        else:
            configuration = DebertaConfig()
        model = AutoModel.from_config(configuration)
        tokenizer = AutoTokenizer.from_pretrained(configuration.name_or_path)
        model.resize_token_embeddings(len(tokenizer))
        folder = f"../data/biosbias/vectors_extracted_from_trained_models/{args.feature_extractor}/random/seed_{args.seed}"
    elif args.model == "finetuning":
        load_path = f"checkpoints/bias_in_bios/{args.feature_extractor}/finetuning/{args.training_data}/{args.training_balanced}/seed_{args.seed}/best_model.pt"
        if args.feature_extractor == 'roberta-base':
            model_ = roBERTa_classifier(N_LABELS)
        else:
            model_ = deBERTa_classifier(N_LABELS)
        load_checkpoint(model_, load_path)
        model = model_.roberta
        folder = f"../data/biosbias/vectors_extracted_from_trained_models/{args.feature_extractor}/{args.model}/{args.training_data}/{args.training_balanced}/seed_{args.seed}"

    return model, folder

def __main__():
    args = parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, folder = get_model(args)
    model.to(device)

    data_path = f"../data/biosbias/tokens_{args.data}_{args.feature_extractor}_128.pt"
    BiasInBios_extract_vectors_data(args.data, model, data_path, feature_extractor=args.feature_extractor, folder=folder)