import argparse
import os
from dataclasses import dataclass
from os.path import join
from typing import List

import torch
from torch import nn
from transformers import set_seed

from mdl import OnlineCodeMDLProbe


@dataclass
class OnlineCodingExperimentResults:
    name: str
    uniform_cdl: float
    online_cdl: float
    compression: float
    report: dict
    fractions: List[float]

def general_probing_args(parser):
    parser.add_argument('--seed', type=int, help='the random seed to check on', required=True)
    parser.add_argument('--model_seed', type=int, help='the random seed used to train the model', required=True)
    parser.add_argument('--embedding_size', type=int, help='embedding size', default=768)
    parser.add_argument('--batch_size', type=int, help='batch size to train the probe', default=16)
    parser.add_argument('--probe_type', type=str, help='linear probe or MLP', choices=['linear', 'mlp'])

def general_MDL_args():
    parser = argparse.ArgumentParser(description='Probe vectors for gender of example using MDL probes.')
    general_probing_args(parser)
    parser.add_argument('--mdl_fractions', nargs='+', type=int, help='linear probe of MLP',
                        default=[2.0, 3.0, 4.4, 6.5, 9.5, 14.0, 21.0, 31.0, 45.7, 67.6, 100])

    return parser

def build_probe(input_size, num_classes=2, probe_type='mlp'):
    probes = {
        'mlp': lambda: nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.Tanh(),
            nn.Linear(input_size // 2, num_classes)
        ),
        'linear': lambda: nn.Linear(input_size, num_classes)
    }
    return probes[probe_type]()

def create_probe(args):
    return build_probe(args.embedding_size, probe_type=args.probe_type)

def run_MDL_probing(args, load_fn, task_name, shuffle):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    online_code_probe = OnlineCodeMDLProbe(lambda: create_probe(args), args.mdl_fractions, device=device)

    train_dataset, val_dataset, test_dataset = load_fn(args)
    train_dataset = train_dataset

    reporting_root = join(os.getcwd(), f'mdl_results/online_coding_{task_name}.pkl')

    uniform_cdl, online_cdl = online_code_probe.evaluate(train_dataset, test_dataset, val_dataset,
                                                         reporting_root=reporting_root, verbose=True, device=device,
                                                         train_batch_size=args.batch_size, shuffle=shuffle)
    compression = round(uniform_cdl / online_cdl, 2)
    report = online_code_probe.load_report(reporting_root)

    exp_results = OnlineCodingExperimentResults(
        name=task_name,
        uniform_cdl=uniform_cdl,
        online_cdl=online_cdl,
        compression=compression,
        report=report,
        fractions=args.mdl_fractions
    )

    return exp_results