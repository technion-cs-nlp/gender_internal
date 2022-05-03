import argparse
from transformers import RobertaTokenizer, DebertaTokenizer
from DataUtils import BiasInBios_extract_tokens_data

parser = argparse.ArgumentParser(description='Extracting tokens for training bias in bios.')
parser.add_argument('--type', type=str, help='type of training data used to train the model',
                    choices=["raw", "scrubbed"], default=None)
parser.add_argument('--model', type=str, help='model_version for tokenizer', choices=['roberta-base', 'deberta-base'])

args = parser.parse_args()

if args.model == 'roberta-base':
    tokenizer = RobertaTokenizer.from_pretrained(args.model)
else:
    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
BiasInBios_extract_tokens_data(args.type, tokenizer, args.model)
