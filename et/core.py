import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import math
from einops import rearrange
from typing import Callable, Optional, Union, Sequence
import pytorch_lightning as pl
from torchnlp.encoders.text import StaticTokenizerEncoder
import constants
from utils import build_embedding_weights
import logging
import numpy as np
from utils import build_optimizer, build_scheduler
from hflayers import HopfieldPooling
from entmax import entmax15, sparsemax, normmax_bisect, entmax_bisect
from torch.nn import TransformerEncoder, TransformerEncoderLayer


TENSOR = torch.Tensor

shell_logger = logging.getLogger(__name__)
functions = {
            "relu" : lambda x: - 0.5 * (F.relu(x) ** 2.0).sum(),
            "softmax": lambda x: -1*torch.logsumexp(x, dim=-1).sum(),
            "tanh": lambda x: -1*torch.log(torch.cosh(x)).sum(),
            "sparsemax": lambda x: -1*(torch.sum(x*sparsemax(x, dim=-1), dim =-1) - torch.norm(sparsemax(x, dim=-1), dim=-1)).sum(),
            "normmax": lambda x: -1*(torch.sum(x*normmax_bisect(x, dim=-1, alpha=5), dim =-1) - torch.norm(normmax_bisect(x, dim=-1, alpha=5,), dim=-1)).sum(),
            "entmax": lambda x: -1*(torch.sum(x*entmax15(x, dim=-1, k=None), dim =-1) - torch.norm(entmax15(x, dim=-1, k=None), dim=-1)).sum()

}
class Lambda(nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, x: TENSOR):
        return self.fn(x)

class EnergyLayerNorm(nn.Module):
    def __init__(self, in_dim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

        self.gamma = nn.Parameter(
            torch.ones(
                1,
            )
        )

        self.bias = nn.Parameter(torch.zeros(in_dim)) if bias else 0.0

    def forward(self, x: TENSOR):
        xu = x.mean(-1, keepdim=True)
        xm = x - xu
        o = xm / torch.sqrt((xm**2.0).mean(-1, keepdim=True) + self.eps)

        return self.gamma * o + self.bias


class PositionEncode(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionEncode, self).__init__()

        self.dropout = nn.Dropout(p=0.1)

        # Compute the positional encodings in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encodings to the input
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Hopfield(nn.Module):
    def __init__(
        self,
        in_dim: int,
        multiplier: float = 4.0,
        fn: Callable = lambda x: -0.5 * (F.relu(x) ** 2.0).sum(),
        bias= False,
    ):
        super().__init__()
        fn = functions[fn]
        self.fn = Lambda(fn)
        self.proj = nn.Linear(in_dim, int(in_dim * multiplier), bias=bias)
        
    def forward(self, g: TENSOR, mask):
        scores = self.proj(g)
        scores[~mask] = -1e6
        x =self.fn(scores)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        in_dim: int,
        qk_dim: int = 64,
        nheads: int = 12,
        beta: Optional[float] = None,
        bias: bool = False,
        att_fn: str = "softmax"
    ):
        super().__init__()
        assert qk_dim > 0 and in_dim > 0
        self.nheads = nheads
        self.h, self.d = nheads, qk_dim
        self.beta = beta if beta is not None else 1.0 / (qk_dim**0.5)
        self.att_fn = functions[att_fn]
        self.wq = nn.Parameter(torch.normal(0, 0.002, size=(nheads, qk_dim, in_dim)))
        self.wk = nn.Parameter(torch.normal(0, 0.002, size=(nheads, qk_dim, in_dim)))

        self.bq = nn.Parameter(torch.zeros(qk_dim)) if bias else None
        self.bk = nn.Parameter(torch.zeros(qk_dim)) if bias else None

    def forward(self, g: TENSOR, mask: Optional[TENSOR] = None):
        q = torch.einsum("...kd, ...hzd -> ...khz", g, self.wq)
        k = torch.einsum("...kd, ...hzd -> ...khz", g, self.wk)

        if self.bq is not None:
            q = q + self.bq
            k = k + self.bk

        # B x H x N x N
        A = torch.einsum("...qhz, ...khz -> ...hqk", q, k)
        expanded_tensor = mask.unsqueeze(1).repeat(1, self.nheads, 1)

        # Expand along the last dimension
        mask = expanded_tensor.unsqueeze(-1).repeat(1, 1, 1, mask.shape[-1])
        mask = mask*mask.transpose(-1, -2)
        A[~mask] = -1e6
        e = (1.0 / self.beta) * self.att_fn(self.beta*A)
        return e

class ETBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        qk_dim: int = 64,
        nheads: int = 12,
        hn_mult: float = 4.0,
        attn_beta: Optional[float] = None,
        attn_bias: bool = False,
        hn_bias: bool = False,
        hn_fn: Callable = lambda x: -0.5 * (F.relu(x) ** 2.0).sum(),
        att_fn: str = "softmax"
    ):
        super().__init__()
        assert qk_dim > 0 and in_dim > 0
        self.hn = Hopfield(in_dim, hn_mult, hn_fn, hn_bias)
        self.attn = Attention(in_dim, qk_dim, nheads, attn_beta, attn_bias, att_fn)

    def energy(
        self,
        g: TENSOR,
        mask: Optional[TENSOR] = None,
    ):
        return self.attn(g, mask) + self.hn(g, mask)

    def forward(
        self,
        g: TENSOR,
        mask: Optional[TENSOR] = None,
    ):
        return self.energy(g, mask)


class ET(pl.LightningModule):
    def __init__(
        self,
        tokenizer: StaticTokenizerEncoder,
        nb_classes: int,
        is_multilabel: bool,
        h_params: dict,
    ):
        out_dim: Optional[int] = h_params.get("out_dim")
        tkn_dim: int = h_params.get("tkn_dim")
        qk_dim: int = h_params.get("qk_dim")
        nheads: int = h_params.get("nheads")
        hn_mult: float = h_params.get("hn_mult")
        attn_beta: Optional[float] = h_params.get("attn_beta")
        attn_bias: bool = h_params.get("attn_bias")
        hn_bias: bool = h_params.get("hn_bias")
        hn_fn: Callable = h_params.get("hn_fn")
        att_fn: Callable = h_params.get("att_fn")
        time_steps: int = h_params.get("time_steps")
        blocks: int = h_params.get("blocks")
        emb_type: int = h_params.get("emb_type")
        emb_path: int = h_params.get("emb_path")
        use_cls: bool = h_params.get("use_cls")
        self.baseline: bool = h_params.get("baseline", False)
        self.alpha: int = h_params.get("alpha")
        super().__init__()
        self.K = time_steps
        self.hparams = h_params
        self.use_cls = use_cls
        self.vocab_size = tokenizer.vocab_size 
        self.is_multilabel = is_multilabel
        self.nb_classes = nb_classes
        
        # define loss function
        criterion_cls = nn.NLLLoss
        # Determine the task based on the number of classes
        task = "multiclass" if nb_classes > 2 else "binary"

        # Initialize metrics
        self.criterion = criterion_cls(reduction="none")
        self.train_accuracy = torchmetrics.Accuracy(task=task,num_classes=nb_classes)
        self.val_accuracy = torchmetrics.Accuracy(task=task, num_classes=nb_classes)
        self.test_accuracy = torchmetrics.Accuracy(task=task,num_classes=nb_classes)
        self.train_precision = torchmetrics.Precision(num_classes=nb_classes, average="macro", task=task)
        self.val_precision = torchmetrics.Precision(num_classes=nb_classes, average="macro", task=task)
        self.test_precision = torchmetrics.Precision(num_classes=nb_classes, average="macro", task=task)
        self.train_recall = torchmetrics.Recall(num_classes=nb_classes, average="macro", task=task)
        self.val_recall = torchmetrics.Recall(num_classes=nb_classes, average="macro", task=task)
        self.test_recall = torchmetrics.Recall(num_classes=nb_classes, average="macro", task=task)
        embedding_weights = build_embedding_weights(
            tokenizer.vocab, emb_type, emb_path, tkn_dim
        )
        self.encode = nn.Embedding(
            self.vocab_size,
            tkn_dim,
            padding_idx=constants.PAD_ID,
            _weight=embedding_weights,
        )

        self.embed_layer = nn.Sequential(self.encode, nn.Dropout(p=0.1))
        self.decode = nn.Sequential(
            nn.LayerNorm(tkn_dim, tkn_dim),
            nn.Linear(tkn_dim, out_dim),
            nn.ReLU()
        )

        self.pos = PositionEncode(tkn_dim) 
        #self.cls = nn.Parameter(torch.ones(1, 1, tkn_dim))
        
        if self.baseline:
            self.transformer_encoder_layer = TransformerEncoderLayer(tkn_dim, nheads, 256)#, dropout=0.1)
            self.transformer_encoder = TransformerEncoder(self.transformer_encoder_layer, 1)
        else:
            self.blocks = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            EnergyLayerNorm(tkn_dim),
                            ETBlock(
                                tkn_dim,
                                qk_dim,
                                nheads,
                                hn_mult,
                                attn_beta,
                                attn_bias,
                                hn_bias,
                                hn_fn,
                                att_fn,
                            ),
                        ]
                    )
                    for _ in range(blocks)
                ]
            )
        self.output_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(out_dim, self.nb_classes),
            nn.LogSoftmax(dim=-1),
        )

    def evolve(
        self,
        x: TENSOR,
        *,
        attn_mask: Optional[Sequence[TENSOR]] = None,
        return_energy: bool = False,
    ):
        energies = [] if return_energy else None

        for norm, et in self.blocks:
            for _ in range(self.K):
                g = norm(x)
                dEdg, E = torch.func.grad_and_value(et)(g, attn_mask)
                x = x - self.alpha * dEdg
                if return_energy:
                    energies.append(E)

        if return_energy:
            g = norm(x)
            E = et(g, attn_mask)
            energies.append(E)

        return x, energies

    def forward(
        self,
        x: TENSOR,
        attn_mask: Optional[Sequence[TENSOR]] = None,
        *,
        return_energy: bool = False,
    ):
        x = self.embed_layer(x)
        x = self.pos(x)
        if self.baseline:
            x = self.transformer_encoder(x , src_key_padding_mask=~attn_mask.t())
        else:
            x, energies = self.evolve(
                x, attn_mask=attn_mask, return_energy=return_energy)
        
        x = self.decode(x)
        if return_energy:
            return self.output_layer(x), energies
        #pooled = self.hopfield_pooling(x)
        pooled = torch.mean(x, dim=1, keepdim=True)
        return self.output_layer(pooled)
    
    def training_step(self, batch: dict, batch_idx: int):
        """
        Compute forward-pass, calculate loss and log metrics.

        :param batch: The dict output from the data module with the following items:
            `input_ids`: torch.LongTensor of shape [B, T],
            `lengths`: torch.LongTensor of shape [B]
            `labels`: torch.LongTensor of shape [B, C]
            `tokens`: list of strings
        :param batch_idx: integer displaying index of this batch
        :return: pytorch_lightning.Result log object
        """
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        mask = input_ids != constants.PAD_ID
        #true_tensor = torch.ones(batch["input_ids"].shape[0], 1, dtype=torch.bool).cuda()
        # cls token
        #mask = torch.cat((true_tensor, mask), dim=1)
        prefix = "train"

        # forward-pass
        y_hat = self(input_ids, attn_mask=mask)

        # compute loss
        y_hat = y_hat if not self.is_multilabel else y_hat.view(-1, self.nb_classes)
        y = labels if not self.is_multilabel else labels.view(-1)
        loss, loss_stats = self.get_loss(y_hat, y, prefix=prefix, mask=mask)

        # logger=False because they are going to be logged via loss_stats
        self.log(
            "train_sum_loss",
            loss.item(),
            prog_bar=True,
            logger=False,
            on_step=True,
            on_epoch=False,
        )

        if self.is_multilabel:
            metrics_to_wandb = {
                "train_loss": loss_stats["criterion"],
            }
        else:
            metrics_to_wandb = {
                "train_loss": loss_stats["mse"],
            }

        self.logger.log_metrics(metrics_to_wandb, self.global_step)

        # return the loss tensor to PTL
        return {"loss": loss}

    def validation_step(self, batch: dict, batch_idx: int):
        output = self._shared_eval_step(batch, batch_idx, prefix="val")
        return output

    def test_step(self, batch: dict, batch_idx: int):
        output = self._shared_eval_step(batch, batch_idx, prefix="test")
        return output

    def _shared_eval_step(self, batch: dict, batch_idx: int, prefix: str):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        mask = input_ids != constants.PAD_ID
        
        #true_tensor = torch.ones(batch["input_ids"].shape[0], 1, dtype=torch.bool).cuda()
        # cls token
        #mask = torch.cat((true_tensor, mask), dim=1)
        y_hat = self(input_ids, attn_mask=mask)

        # compute loss
        y_hat = y_hat if not self.is_multilabel else y_hat.view(-1, self.nb_classes)
        y = labels if not self.is_multilabel else labels.view(-1)
        loss, loss_stats = self.get_loss(y_hat, y, prefix=prefix, mask=mask)

        self.logger.agg_and_log_metrics(loss_stats, step=None)
        self.log(
            f"{prefix}_sum_loss",
            loss.item(),
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        # output to be stacked across iterations
        output = {
            f"{prefix}_sum_loss": loss.item(),
            f"{prefix}_tokens": batch["tokens"],
            f"{prefix}_predictions": y_hat,
            f"{prefix}_labels": batch["labels"].tolist(),
            f"{prefix}_lengths": batch["lengths"].tolist(),
        }

        if "mse" in loss_stats.keys():
            output[f"{prefix}_mse"] = loss_stats["mse"]

        return output

    def training_epoch_end(self, outputs: list):
        """
        PTL hook.

        :param outputs: list of dicts representing the stacked outputs from training_step
        """
        print("\nEpoch ended.\n")

    def validation_epoch_end(self, outputs: list):
        self._shared_eval_epoch_end(outputs, prefix="val")

    def test_epoch_end(self, outputs: list):
        self._shared_eval_epoch_end(outputs, prefix="test")

    def _shared_eval_epoch_end(self, outputs: list, prefix: str):
        """
        PTL hook. Perform validation at the end of an epoch.

        :param outputs: list of dicts representing the stacked outputs from validation_step
        :param prefix: `val` or `test`
        """
        # assume that `outputs` is a list containing dicts with the same keys
        stacked_outputs = {k: [x[k] for x in outputs] for k in outputs[0].keys()}

        # average across batches
        avg_outputs = {
            f"avg_{prefix}_sum_loss": np.mean(stacked_outputs[f"{prefix}_sum_loss"]),
        }

        shell_logger.info(
            f"Avg {prefix} sum loss: {avg_outputs[f'avg_{prefix}_sum_loss']:.4}"
        )

        dict_metrics = {
            f"avg_{prefix}_sum_loss": avg_outputs[f"avg_{prefix}_sum_loss"],
        }


        # log classification metrics
        if self.is_multilabel:
            preds = torch.argmax(
                torch.cat(stacked_outputs[f"{prefix}_predictions"]), dim=-1
            )
            labels = torch.tensor(
                [
                    item
                    for sublist in stacked_outputs[f"{prefix}_labels"]
                    for item in sublist
                ],
                device=preds.device,
            )
            if prefix == "val":
                accuracy = self.val_accuracy(preds, labels)
                precision = self.val_precision(preds, labels)
                recall = self.val_recall(preds, labels)
                f1_score = 2 * precision * recall / (precision + recall)
            else:
                accuracy = self.test_accuracy(preds, labels)
                precision = self.test_precision(preds, labels)
                recall = self.test_recall(preds, labels)
                f1_score = 2 * precision * recall / (precision + recall)

            dict_metrics[f"{prefix}_precision"] = precision
            dict_metrics[f"{prefix}_recall"] = recall
            dict_metrics[f"{prefix}_f1score"] = f1_score
            dict_metrics[f"{prefix}_accuracy"] = accuracy

            shell_logger.info(f"{prefix} accuracy: {accuracy:.4}")
            shell_logger.info(f"{prefix} precision: {precision:.4}")
            shell_logger.info(f"{prefix} recall: {recall:.4}")
            shell_logger.info(f"{prefix} f1: {f1_score:.4}")

            self.log(
                f"{prefix}_f1score",
                dict_metrics[f"{prefix}_f1score"],
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
            )

        else:
            avg_outputs[f"avg_{prefix}_mse"] = np.mean(stacked_outputs[f"{prefix}_mse"])
            shell_logger.info(
                f"Avg {prefix} MSE: {avg_outputs[f'avg_{prefix}_mse']:.4}"
            )
            dict_metrics[f"avg_{prefix}_mse"] = avg_outputs[f"avg_{prefix}_mse"]

            self.log(
                f"{prefix}_MSE",
                dict_metrics[f"avg_{prefix}_mse"],
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
            )

        self.logger.agg_and_log_metrics(dict_metrics, self.current_epoch)

        self.log(
            f"avg_{prefix}_sum_loss",
            dict_metrics[f"avg_{prefix}_sum_loss"],
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        if self.is_multilabel:
            output = {
                f"avg_{prefix}_sum_loss": dict_metrics[f"avg_{prefix}_sum_loss"],
                f"{prefix}_precision": precision,
                f"{prefix}_recall": recall,
                f"{prefix}_f1score": f1_score,
                f"{prefix}_accuracy": accuracy,
            }
        else:
            output = {
                f"avg_{prefix}_sum_loss": dict_metrics[f"avg_{prefix}_sum_loss"],
                f"avg_{prefix}_MSE": dict_metrics[f"avg_{prefix}_mse"],
            }

        return output

    def configure_optimizers(self):
        """Configure optimizers and lr schedulers for Trainer."""
        optimizer = build_optimizer(self.parameters(), self.hparams)
        scheduler = build_scheduler(optimizer, self.hparams)
        output = {"optimizer": optimizer}
        if scheduler is not None:
            output["scheduler"] = scheduler
            # output["monitor"] = self.criterion  # not sure we need this
        return output
    
    def get_loss(self, y_hat, y, prefix, mask=None):
        """
        :param y_hat: predictions from SentimentPredictor. Torch.Tensor of shape [B, C]
        :param y: tensor with gold labels. torch.BoolTensor of shape [B]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :return: tuple containing:
            `loss cost (torch.FloatTensor)`: the result of the loss function
            `loss stats (dict): dict with loss statistics
        """
        stats = {}
        loss_vec = self.criterion(y_hat, y)  # [B] or [B,C]
        loss = loss_vec.mean()  # [1]
        stats["criterion"] = float(loss.item())  # [1]

        return loss, stats