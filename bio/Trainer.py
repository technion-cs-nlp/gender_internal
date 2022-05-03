import abc
import json
from collections import Iterator, defaultdict
from abc import abstractmethod, ABC
from typing import NamedTuple

import numpy as np
import torch
import wandb
from allennlp.fairness import Independence, Separation, Sufficiency
from sklearn.neural_network import MLPClassifier
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

from tqdm import tqdm
from Data import Data, BiasInBiosDataFinetuning, BiasInBiosDataLM, BiasInBiosData



class BestModel(NamedTuple):
    state_dict: dict
    epoch: int
    result: dict

def load_checkpoint(model, load_path):

    if load_path == None:
        return

    state_dict = torch.load(load_path)
    print(f'Model loaded from <== {load_path}')

    if callable(state_dict['model_state_dict']):
        model.load_state_dict(state_dict['model_state_dict']())
    else:
        model.load_state_dict(state_dict['model_state_dict'])

class Trainer(ABC):
    """
        A class abstracting the various tasks of training models.

        Provides methods at multiple levels of granularity:
        - Multiple epochs (fit)
        - Single epoch (train_epoch/test_epoch)
        - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, bsz, scheduler=None, device='cpu', seed=None):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.verbose = False
        self.best_model: BestModel = None
        self.metric = 'acc'
        self.batch_size = bsz
        self.seed = seed

        # move to device
        self.model.to(self.device)

    def fit(self, train_data: Data, val_data: Data, num_epochs, checkpoint_folder, print_every=1,
            checkpoint_every=1):

        for epoch in range(1, num_epochs + 1):

            self.verbose = ((epoch % print_every) == 0 or (epoch == num_epochs - 1))
            self._print(f'--- EPOCH {epoch}/{num_epochs} ---', self.verbose)

            self.model.train()
            train_result = self.train_epoch(train_data)
            self.model.eval()
            valid_result = self.evaluate(val_data, "valid")

            self.best_checkpoint(valid_result, epoch)
            if (epoch % checkpoint_every) == 0:
                self.save_checkpoint(checkpoint_folder, epoch, valid_result, save_best=False)

            # print(train_result, valid_result)
            self.log_epoch_results(train_result, valid_result)

        self.save_checkpoint(checkpoint_folder, None, None, save_best=True)

    @abstractmethod
    def train_epoch(self, train_data: Data):
        """
                Train once over a training set (single epoch).
                :param train_data: the training data object
                :return: An epoch result dictionary.
                """
        ...

    def train_batch(self, batch):

        self.optimizer.zero_grad()
        res = self.forward_fn(batch)
        loss = res['loss']
        loss.backward()
        # grads = []
        # for p in self.model.parameters():
        #     grads.append(torch.norm(p.grad).item())
        # wandb.log({"avg grad": np.mean(grads), "batch loss": loss.item()})
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()
            # lr = self.scheduler.get_last_lr()
            # print(lr)

        return res

    def save_checkpoint(self, save_folder, epoch, valid_result, save_best):

        if save_folder == None:
            return

        Path(save_folder).mkdir(parents=True, exist_ok=True)

        if not save_best:
            save_path = f"{save_folder}/ckpt_epoc_{epoch}.pt"
            torch.save({'model_state_dict': self.model.state_dict(),
                        'epoch_data': valid_result,
                        'epoch': epoch}, save_path)
        else:
            save_path = f"{save_folder}/best_model.pt"
            torch.save({'model_state_dict': self.best_model.state_dict,
                        'epoch_data': self.best_model.result,
                        'epoch': self.best_model.epoch}, save_path)

        print(f'Model saved to ==> {save_path}')

    def load_checkpoint(self, load_path):

        if load_path == None:
            return

        state_dict = torch.load(load_path)
        print(f'Model loaded from <== {load_path}')

        if callable(state_dict['model_state_dict']):
            self.model.load_state_dict(state_dict['model_state_dict']())
        else:
            self.model.load_state_dict(state_dict['model_state_dict'])

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    def best_checkpoint(self, valid_result, epoch):
        if (not self.best_model) or self.best_model.result[self.metric] < valid_result[self.metric]:
            self.best_model = BestModel(self.model.state_dict(), epoch, valid_result.copy())
            wandb.run.summary["best_metric"] = valid_result[self.metric]
            wandb.run.summary["best_epoch"] = epoch
            wandb.run.summary["metric"] = self.metric

    @abstractmethod
    def forward_fn(self, batch):
        ...

    @abstractmethod
    def evaluate(self, data: Data, split_type: str):
        ...

    @abstractmethod
    def log_epoch_results(self, train_result, valid_result):
        ...


class BiasInBiosTrainer(Trainer):

    def log_epoch_results(self, train_result, valid_result):

        train_result_new = {}
        for k in train_result:
            train_result_new[f"train_{k}"] = train_result[k]
        valid_result_new = {}
        for k in valid_result:
            valid_result_new[f"valid_{k}"] = valid_result[k]

        wandb.log({**train_result_new, **valid_result_new})

    def evaluate(self, data: Data, split_type: str):
        dl = DataLoader(data.dataset, batch_size=self.batch_size, shuffle=False)

        y_pred, logits = self.predict(dl)
        y = data.dataset.tensors[1].to(self.device)
        loss = self.loss_fn(logits, y)
        total_correct = torch.sum(y == logits.argmax(dim=1)).item()

        total_examples = len(y)
        accuracy = total_correct / total_examples
        perc = self.get_perc(data)
        gap_res = self.gap(data, y_pred, split_type, perc)
        fairness = self.allennlp_metrics(data, y_pred, perc)

        return {"loss": loss, "acc": accuracy, **gap_res, **fairness}

    def predict(self, data: DataLoader):
        self.model.eval()
        all_y_pred = []
        all_logits = []

        with torch.no_grad():
            for batch in tqdm(data):
                logits = self.forward_fn(batch)['logits']
                y_pred = logits.argmax(dim=1)
                all_y_pred.append(y_pred)
                all_logits.append(logits)

        all_y_pred = torch.cat(all_y_pred, dim=0)
        all_logits = torch.cat(all_logits, dim=0)
        return all_y_pred, all_logits

    def train_epoch(self, train_data: Data):
        losses = []
        total_correct = 0
        total_examples = 0

        train_iter = DataLoader(train_data.dataset, batch_size=self.batch_size, shuffle=True)

        for batch in tqdm(train_iter):

            y = batch[1]
            batch_res = self.train_batch(batch)
            losses.append(batch_res['loss'].item())
            total_correct += batch_res['n_correct']
            total_examples += len(y)

        accuracy = total_correct / total_examples
        loss = np.mean(losses)

        return {"loss": loss, "acc": accuracy}

    def get_perc(self, data):

        perc_dict = data.perc

        label_to_code = data.get_label_to_code()
        perc = []
        for i, (profession, label) in enumerate(sorted(label_to_code.items(),  key=lambda item: item[1])):

            perc.append(perc_dict[profession])

        return perc

    def gap(self, data: BiasInBiosDataFinetuning, y_pred, split_type, perc):

        tpr_gap = []
        fpr_gap = []
        precision_gap = []
        F1_gap = []

        golden_y = data.dataset.tensors[1]


        z = data.z

        for label in torch.unique(golden_y):

            m_res = self.metrics_fn(y_pred, golden_y, z, label.item(), 'M')
            f_res = self.metrics_fn(y_pred, golden_y, z, label.item(), 'F')

            tpr_gap.append(f_res["tpr"] - m_res["tpr"])
            fpr_gap.append(f_res["fpr"] - m_res["fpr"])
            precision_gap.append(f_res["precision"] - m_res["precision"])
            F1_gap.append(f_res["f1_score"] - m_res["f1_score"])

        return {"tpr_gap": tpr_gap, "pearson_tpr_gap": np.corrcoef(perc, tpr_gap)[0, 1],
                "fpr_gap": fpr_gap, "pearson_fpr_gap": np.corrcoef(perc, fpr_gap)[0, 1],
                "precision_gap": precision_gap, "pearson_precision_gap": np.corrcoef(perc, precision_gap)[0, 1],
                "F1_gap": F1_gap, "pearson_F1_gap": np.corrcoef(perc, F1_gap)[0, 1],
                "mean abs tpr gap": np.abs(tpr_gap).mean(),
                "mean abs fpr gap": np.abs(fpr_gap).mean(),
                "mean abs f1 gap": np.abs(F1_gap).mean(),
                "mean abs precision gap": np.abs(precision_gap).mean(),
                "perc": perc,
                }

    def metrics_fn(self, y_pred: torch.Tensor, golden_y, z, label: int, gender: str):
        assert (len(y_pred) == len(golden_y))

        tp_indices = np.logical_and(np.logical_and(z == gender, golden_y.cpu() == label),
                                    y_pred.cpu() == label).int()  # only correct predictions of this gender

        n_tp = torch.sum(tp_indices).item()
        pos_indices = np.logical_and(z == gender, golden_y.cpu() == label).int()
        n_pos = torch.sum(pos_indices).item()
        tpr = n_tp / n_pos

        fp_indices = np.logical_and(np.logical_and(z == gender, golden_y.cpu() != label), y_pred.cpu() == label).int()
        neg_indices = np.logical_and(z == gender, golden_y.cpu() != label).int()
        n_fp = torch.sum(fp_indices).item()
        n_neg = torch.sum(neg_indices).item()
        fpr = n_fp / n_neg

        n_total_examples = len(y_pred)
        precision = n_tp / n_total_examples

        if precision * tpr == 0:
            f1_score = 0
        else:
            f1_score = 2 * ((precision * tpr) / (precision + tpr))

        return {"tpr": tpr, "fpr": fpr, "precision": precision, "f1_score": f1_score}

    def allennlp_metrics(self, data: BiasInBiosData, y_pred, perc):

        z = data.z.copy()
        z[z == 'M'] = 0
        z[z == 'F'] = 1
        z = torch.Tensor(z.astype(int))
        y = data.dataset.tensors[1].cpu()
        y_pred = y_pred.cpu()

        independence = Independence(data.n_labels, 2)
        independence(y_pred, z)
        independence_score = independence.get_metric()

        separation = Separation(data.n_labels, 2)
        separation(y_pred, y, z)
        separation_score = separation.get_metric()

        sufficiency = Sufficiency(data.n_labels, 2, dist_metric="wasserstein")
        sufficiency(y_pred, y, z)
        sufficiency_score = sufficiency.get_metric()

        self.dictionary_torch_to_number(independence_score)
        self.dictionary_torch_to_number(separation_score)
        self.dictionary_torch_to_number(sufficiency_score)

        separation_gaps = [scores[0] - scores[1] for label, scores in sorted(separation_score.items())] # positive value - more separation for women
        # pearson_separation_gaps = np.corrcoef(perc, separation_gaps)[0, 1]
        sufficiency_gaps = [scores[0] - scores[1] for label, scores in sorted(sufficiency_score.items())]
        # pearson_sufficiency_gaps = np.corrcoef(perc, sufficiency_gaps)[0, 1]

        return {"independence": json.dumps(independence_score), "separation": json.dumps(separation_score), "sufficiency": json.dumps(sufficiency_score),
                "independence_sum": independence_score[0] + independence_score[1],
                "separation_gaps": separation_gaps,
                "sufficiency_gaps": sufficiency_gaps}

    def dictionary_torch_to_number(self, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                self.dictionary_torch_to_number(v)
            else:
                d[k] = v.item()

class FinetuningBiasInBiosTrainer(BiasInBiosTrainer):

    def forward_fn(self, batch):
        batch = tuple(t.to(self.device) for t in batch)
        X, y, att_mask = batch

        logits = self.model.forward(X, att_mask)  # shape (batch_size, n_labels)
        loss = self.loss_fn(logits, y)
        n_correct = torch.sum(y == logits.argmax(dim=1)).item()

        return {'loss': loss, 'n_correct': n_correct, 'logits': logits}

class NoFinetuningBiasInBiosTrainer(BiasInBiosTrainer):

    def forward_fn(self, batch):
        batch = tuple(t.to(self.device) for t in batch)
        X, y = batch  # X shape (bsz, 768)

        logits = self.model.forward(X)  # shape (bsz, n_labels)
        loss = self.loss_fn(logits, y)
        n_correct = torch.sum(y == logits.argmax(dim=1)).item()

        return {'loss': loss, 'n_correct': n_correct, 'logits': logits}
