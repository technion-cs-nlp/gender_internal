import abc
import string
from typing import Union, List

import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from abc import ABC, abstractmethod

from DataUtils import BiasinBios_split_and_save_vectors_data, BiasinBios_split_and_return_vectors_data, balance_dataset, \
    BiasinBios_split_and_return_tokens_data


class Data(ABC):BiasInBios_extract_vectors.py
    """
    An abstract class for loading and arranging data from files.
    """

    def __init__(self):
        self.dataset = None

class BiasInBiosData(Data):
    def __init__(self, path: Union[str, List[str]], seed, split, balanced):
        super().__init__()
        self.dataset = None
        self.original_y = None
        self.original_z = None
        self.n_labels = 0
        self.z = None
        self.load_bias_in_bios_dataset(path, seed, split, balanced)
        self.perc = self.compute_perc()

    @abstractmethod
    def load_bias_in_bios_dataset(self, path : str, seed, split, balanced=None):
        ...

    def get_label_to_code(self):
        code_to_label = dict(enumerate(self.cat.categories))
        label_to_code = {v: k for k, v in code_to_label.items()}
        return label_to_code

    def compute_perc(self):
        perc = {}
        golden_y = self.original_y
        for profession in np.unique(golden_y):
            total_of_label = len(golden_y[golden_y == profession])
            indices_female = np.logical_and(golden_y == profession, self.original_z == 'F')
            perc_female = len(golden_y[indices_female]) / total_of_label
            perc[profession] = perc_female

        return perc

class BiasInBiosDataFinetuning(BiasInBiosData):

    def split_and_balance(self, path, seed, split, balanced=None, other=None):
        data = BiasinBios_split_and_return_tokens_data(seed, path, other=other)
        self.cat = data["categories"]

        X, y, masks, other = data[split]["X"], data[split]["y"], data[split]["masks"], data[split]["other"]

        self.z = data[split]["z"]
        self.original_z = data[split]["z"]

        if balanced in ("oversampled", "subsampled"):
            output = balance_dataset(X, y, self.z, masks=masks, other=other,
                                                  oversampling=True if balanced == "oversampled" else False)
            X, y, self.z, masks = output[0], output[1], output[2], output[3]
            if other is not None:
                other = output[4]

        self.cat = data["categories"]
        y = torch.tensor(y).long()
        X = torch.tensor(X).long()
        masks = torch.tensor(masks).long()

        self.original_y = data[split]["original_y"]
        self.n_labels = len(np.unique(self.original_y))

        if other is not None:
            return X, y, masks, other
        else:
            return X, y, masks

    def load_bias_in_bios_dataset(self, path : str, seed, split, balanced=None):

        X, y, masks = self.split_and_balance(path, seed, split, balanced)

        self.dataset = TensorDataset(X, y, masks)

class BiasInBiosDataLinear(BiasInBiosData):

    def load_bias_in_bios_dataset(self, path : str, seed, split, balanced=None):
        data = BiasinBios_split_and_return_vectors_data(seed, path)
        self.cat = data["categories"]
        X, y = data[split]["X"], data[split]["y"]

        self.z = data[split]["z"]
        self.original_z = data[split]["z"]

        if balanced in ("oversampled", "subsampled"):
            X, y, self.z = balance_dataset(X, y, self.z, oversampling=True if balanced=="oversampled" else False)

        y = torch.tensor(y).long()
        X = torch.tensor(X)

        self.dataset = TensorDataset(X, y)
        self.original_y = data[split]["original_y"]
        self.n_labels = len(np.unique(self.original_y))
