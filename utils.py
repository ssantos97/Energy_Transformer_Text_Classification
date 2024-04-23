import csv
import gzip
import logging
import os
import pickle
import random
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from pytorch_lightning import seed_everything
from pytorch_lightning.core.saving import load_hparams_from_tags_csv
from pytorch_lightning.loggers import WandbLogger
from torchnlp.encoders.text import StaticTokenizerEncoder
from torchnlp.word_to_vector import GloVe

import constants
import json
import pickle
from pathlib import Path
from copy import deepcopy
import yaml

def build_embedding_weights(vocab: dict, emb_type: str, emb_path: str, emb_size: int):
    if emb_type == "glove":
        return load_glove_embeddings(vocab, emb_path, emb_size)
    # nn.Embedding will initialize the weights randomly
    print("Random weights will be used as embeddings.")
    return None

class Grid:
    """
    Specifies the configuration for multiple models.
    """

    def __init__(self, path_or_dict):
        self.configs_dict = read_config_file(path_or_dict)
        # Convert 'None' strings to Python None
        for key, value in self.configs_dict.items():
            if isinstance(value[0], str) and value[0].lower() == 'none':
                self.configs_dict[key] = [None]
        self.num_configs = 0  # must be computed by _create_grid
        self._configs = self._create_grid()

    def __getitem__(self, index):
        return self._configs[index]

    def __len__(self):
        return self.num_configs

    def __iter__(self):
        assert self.num_configs > 0, 'No configurations available'
        return iter(self._configs)

    def _grid_generator(self, cfgs_dict):
        keys = cfgs_dict.keys()
        result = {}

        if cfgs_dict == {}:
            yield {}
        else:
            configs_copy = deepcopy(cfgs_dict)  # create a copy to remove keys

            # get the "first" key
            param = list(keys)[0]
            del configs_copy[param]

            first_key_values = cfgs_dict[param]
            for value in first_key_values:
                result[param] = value

                for nested_config in self._grid_generator(configs_copy):
                    result.update(nested_config)
                    yield deepcopy(result)

    def _create_grid(self):
        '''
        Takes a dictionary of key:list pairs and computes all possible permutations.
        :param configs_dict:
        :return: A dictionary generator
        '''
        config_list = [cfg for cfg in self._grid_generator(self.configs_dict)]
        self.num_configs = len(config_list)
        return config_list

def read_config_file(dict_or_filelike):
    if isinstance(dict_or_filelike, dict):
        return dict_or_filelike

    path = Path(dict_or_filelike)
    if path.suffix == ".json":
        return json.load(open(path, "r"))
    elif path.suffix in [".yaml", ".yml"]:
        return yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    elif path.suffix in [".pkl", ".pickle"]:
        return pickle.load(open(path, "rb"))

    raise ValueError("Only JSON, YaML and pickle files supported.")

def build_optimizer(model_parameters, h_params: dict):
    # get valid parameters (unfreezed ones)
    # parameters = filter(lambda p: p.requires_grad, model_parameters)
    parameters = model_parameters
    if h_params["optimizer"] == "adam":
        return torch.optim.Adam(
            parameters,
            lr=h_params["lr"],
            betas=h_params["betas"],
            weight_decay=h_params["weight_decay"],
            amsgrad=h_params["amsgrad"],
        )
    elif h_params["optimizer"] == "adadelta":
        return torch.optim.Adadelta(
            parameters,
            lr=h_params["lr"],
            rho=h_params["rho"],
            weight_decay=h_params["weight_decay"],
        )
    elif h_params["optimizer"] == "adadelta":
        return torch.optim.Adagrad(
            parameters, lr=h_params["lr"], weight_decay=h_params["weight_decay"]
        )
    elif h_params["optimizer"] == "adamax":
        return torch.optim.Adamax(
            parameters,
            lr=h_params["lr"],
            betas=h_params["betas"],
            weight_decay=h_params["weight_decay"],
        )
    elif h_params["optimizer"] == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=h_params["lr"],
            betas=h_params["betas"],
            weight_decay=h_params["weight_decay"],
            amsgrad=h_params["amsgrad"],
        )
    elif h_params["optimizer"] == "sparseadam":
        return torch.optim.SparseAdam(
            parameters,
            lr=h_params["lr"],
            betas=h_params["betas"],
        )
    elif h_params["optimizer"] == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=h_params["lr"],
            momentum=h_params["momentum"],
            dampening=h_params["dampening"],
            weight_decay=h_params["weight_decay"],
            nesterov=h_params["nesterov"],
        )
    elif h_params["optimizer"] == "asgd":
        return torch.optim.ASGD(
            parameters,
            lr=h_params["lr"],
            lambd=h_params["lambd"],
            alpha=h_params["alpha"],
            t0=h_params["t0"],
            weight_decay=h_params["weight_decay"],
        )
    elif h_params["optimizer"] == "rmsprop":
        return torch.optim.RMSprop(
            parameters,
            lr=h_params["lr"],
            alpha=h_params["alpha"],
            weight_decay=h_params["weight_decay"],
            momentum=h_params["momentum"],
            centered=h_params["centered"],
        )
    else:
        raise Exception(f"Optimizer `{h_params['optimizer']}` not available.")


def build_scheduler(optimizer: torch.optim.Optimizer, h_params: dict):
    """Returns a torch lr_scheduler object or None in case a scheduler is not specified."""
    if "scheduler" not in h_params or h_params["scheduler"] is None:
        return None
    elif h_params["scheduler"] == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=h_params["step_size"], gamma=h_params["lr_decay"]
        )
    elif h_params["scheduler"] == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=h_params["milestones"], gamma=h_params["lr_decay"]
        )
    elif h_params["scheduler"] == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=h_params["lr_decay"]
        )
    elif h_params["scheduler"] == "cosine-annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=h_params["T_max"], eta_min=h_params["eta_min"]
        )
    elif h_params["scheduler"] == "cosine-annealing":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=h_params["T_0"],
            T_mult=h_params["T_mult"],
            eta_min=h_params["eta_min"],
        )
    elif h_params["scheduler"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=h_params["lr_decay"],
            patience=h_params["patience"],
            cooldown=h_params["cooldown"],
            threshold=h_params["threshold"],
            min_lr=h_params["min_lr"],
        )
    else:
        raise Exception(f"Scheduler `{h_params['scheduler']}` not available.")

def configure_output_dir(output_dir: str):
    """
    Create a directory (recursively) and ignore errors if they already exist.

    :param output_dir: path to the output directory
    :return: output_path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def configure_seed(seed: int):
    """
    Seed everything: python, random, numpy, torch, torch.cuda.

    :param seed: seed integer (if None, a random seed will be created)
    :return: seed integer
    """
    seed = seed_everything(seed)
    return seed


def configure_shell_logger(output_dir: str):
    """Configure logger with a proper log format and save log to a file."""
    log_format = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    if output_dir is not None:
        fh = logging.FileHandler(os.path.join(output_dir, "out.log"))
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)


def save_object(obj: object, path: str):
    """
    Dump an object (e.g. tokenizer or label encoder) via pickle.

    :param obj: any object (e.g. pytorch-nlp's tokenizer instance)
    :param path: path to save the object
    """
    with open(path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_config_to_csv(dict_args: dict, path: str):
    """
    Save the meta data config to csv in the run folder as "meta_tags.csv"

    :param obj: dict with the data
    :param path: path to save the object
    """
    meta_tags_path = os.path.join(path, "meta_tags.csv")
    if not os.path.exists(path):
        os.mkdir(path)
    with open(meta_tags_path, "w") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dict_args.items():
            writer.writerow([key, value])


def load_object(path: str):
    """
    Unpickle a saved object.

    :param path: path to a pickled object
    :return: the object
    """
    with open(path, "rb") as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


def load_yaml_config(path: str):
    """
    From: https://github.com/joeynmt/joeynmt/

    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dict
    """
    with open(path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def setup_wandb_logger(default_root_dir: str):
    """
    Function that sets the WanbLogger to be used.

    :param default_root_dir: logs save dir.
    """
    id = uuid.uuid4()
    return WandbLogger(
        project="ET",
        save_dir=default_root_dir,
        version=str(id.fields[1]),
    )


def find_last_checkpoint_version(path_to_logs: str):
    """Sort the log directory to pick the last timestamped checkpoint filename."""

    def get_time_from_version_name(name: str):
        # name format example `version_16-10-2020_08-12-48`
        timestamp = name[6:]
        return timestamp

    ckpt_versions = os.listdir(path_to_logs)
    if len(ckpt_versions) == 0:
        return None
    ckpt_versions.sort(key=get_time_from_version_name)

    ckpt_dir = os.path.join(path_to_logs, ckpt_versions[-1], "checkpoints/")
    ckpt_epochs = os.listdir(ckpt_dir)
    if len(ckpt_epochs) == 0:
        return None
    ckpt_epochs.sort(key=lambda x: int(x[6:].split(".")[0]))  # e.g. epoch=2.ckpt

    return os.path.join(ckpt_dir, ckpt_epochs[-1])


def load_ckpt_config(ckpt_path: str):
    """
    Load the .csv config file stored with the checkpoint and transform it to a dict object.
    :param ckpt_path: path to a saved checkpoint.
    :return: config dict
    """
    csv_config_dir = os.path.dirname(os.path.dirname(ckpt_path))
    csv_config_path = os.path.join(csv_config_dir, "meta_tags.csv")
    config_dict = load_hparams_from_tags_csv(csv_config_path)
    return config_dict



def load_glove_embeddings(vocab: list, name: str, emb_size: int):
    """
    Load pre-trained Glove embeddings using PyTorch-NLP interface:
    https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.word_to_vector.html

    :param vocab: list of tokens
    :param name: Glove name version (e.g. ‘840B’, ‘twitter.27B’, ‘6B’, ‘42B’)
    :param emb_size: word embedding size
    :return: Torch.FloatTensor with shape (vocab_size, emb_dim)
    """
    vocab_set = set(vocab)
    unk_vector = torch.FloatTensor(emb_size).uniform_(-0.05, 0.05)
    unk_init = lambda v: unk_vector
    pretrained_embedding = GloVe(
        name=name,
        dim=emb_size,
        unk_init=unk_init,
        is_include=lambda w: w in vocab_set,
    )
    embedding_weights = torch.FloatTensor(len(vocab), emb_size)
    for idx, token in enumerate(vocab):
        if token in [constants.PAD, constants.SOS, constants.EOS]:
            if token in pretrained_embedding.token_to_index:
                embedding_weights[idx] = pretrained_embedding[token]
            else:
                if token == constants.PAD:  # zero vector for padding
                    embedding_weights[idx] = torch.zeros(emb_size)
                else:  # random token for everything else
                    embedding_weights[idx] = torch.FloatTensor(emb_size).uniform_(
                        -0.05, 0.05
                    )
        else:
            embedding_weights[idx] = pretrained_embedding[token]
    return embedding_weights


def unroll(list_of_lists, rec=False):
    """
    Unroll a list of lists
    Args:
        list_of_lists (list): a list that contains lists
        rec (bool): unroll recursively
    Returns:
        a single list
    """
    if not isinstance(list_of_lists[0], (np.ndarray, list, torch.Tensor)):
        return list_of_lists
    new_list = [item for ell in list_of_lists for item in ell]
    if rec and isinstance(new_list[0], (np.ndarray, list, torch.Tensor)):
        return unroll(new_list, rec=rec)
    return new_list