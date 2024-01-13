import abc
import numbers
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.nn import *
from torch import Tensor
from torch.distributions import Normal
from torch.nn.modules.normalization import _shape_t
from lightning.pytorch.utilities import grad_norm
from tqdm import tqdm

###############################################################################
# Layer Definition


class MonteCarloDropout(nn.Dropout):
    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, True, self.inplace)


class PositionEmbedding(nn.Module):
    def __init__(
        self, max_length: int, embedding_size: int, dropout: float, inplace: bool
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout, inplace)
        self.position_embedding = nn.Embedding(max_length, embedding_size)

    def forward(self, input: Tensor) -> Tensor:
        position_idx = torch.arange(input.shape[0], device=input.device).unsqueeze(1)
        position_embeddings = self.position_embedding(position_idx)
        return self.dropout(input + position_embeddings)


class ReeoLayerNorm(nn.Module):
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        bias: bool = True,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            self.bias = (
                Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
                if bias
                else None
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )

    def extra_repr(self) -> str:
        return (
            "{normalized_shape}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


###############################################################################
# Model Definition
class Model(pl.LightningModule, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def _shared_step(self, batch, phase: str) -> Dict[str, Tensor]:
        pass

    @staticmethod
    def _get_prefix(phase: str) -> str:
        if phase == "train":
            return ""
        elif phase == "val" or phase == "test":
            return phase + "_"
        else:
            raise ValueError(
                f"Expect phase is either train, val, or test. Got {phase} instead."
            )

    def _replace(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                setattr(
                    module,
                    name,
                    nn.Linear(child.in_features, child.out_features, self.hparams.bias),
                )
            elif isinstance(child, nn.MultiheadAttention):
                setattr(
                    module,
                    name,
                    nn.MultiheadAttention(
                        self.hparams.hidden_size,
                        self.hparams.num_heads,
                        dropout=self.hparams.dropout,
                        batch_first=self.hparams.batch_first,
                        bias=self.hparams.bias,
                    ),
                )
            elif isinstance(child, nn.LayerNorm):
                setattr(
                    module,
                    name,
                    ReeoLayerNorm(child.normalized_shape, bias=self.hparams.bias),
                )
            elif isinstance(child, nn.Dropout):
                setattr(module, name, nn.Dropout(child.p, inplace=self.hparams.inplace))
            elif list(child.named_children()):
                self._replace(child)
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_before_zero_grad(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms, on_step=False, on_epoch=True, prog_bar=False, logger=True)


class NSNET2(Model):
    def __init__(self) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.ff1 = nn.Linear(255, 400)
        self.gru = nn.GRU(400, 400, 2)
        self.ff2 = nn.Linear(400, 600)
        self.ff3 = nn.Linear(600, 600)
        self.ff4 = nn.Linear(600, 255)
    
    def forward(self, inputs: Tensor) -> Tensor:
        outputs = F.relu(self.ff1(inputs))
        outputs, _ = self.gru(outputs)
        outputs = F.relu(self.ff2(outputs))
        outputs = F.relu(self.ff3(outputs))
        outputs = F.relu(self.ff4(outputs))
        return outputs

    def _shared_step(self, batch, phase: str) -> Dict[str, Tensor]:
        noisy_audios, clean_audios = batch
        filters = self.forward(noisy_audios)
        filtered_audios = noisy_audios * filters
        loss = F.mse_loss(filtered_audios, clean_audios)
        self.log(
            f"{phase}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss


# class ProbVerticalStackReeoModule(Model):
#     def __init__(
#         self,
#         sequence_length: int,
#         audio_feature_size: int,
#         motion_feature_size: int,
#         hidden_size: int,
#         num_heads: int,
#         feedforward_size: int,
#         dropout: float,
#         inplace: bool,
#         activation: str,
#         num_layers: int,
#         batch_first: bool,
#         norm_first: bool,
#         bias: bool,
#         step: int,
#         noise_prob: float,
#         noise_scale: float,
#         zero_initial_prob: float,
#     ) -> None:
#         super().__init__()
#         self.save_hyperparameters()
#         self.pos_embedding = PositionEmbedding(
#             sequence_length, hidden_size, 0.1, inplace
#         )
#         self.linear_emdedding = nn.Linear(
#             audio_feature_size + motion_feature_size, hidden_size, bias
#         )
#         layer = nn.TransformerEncoderLayer(
#             d_model=hidden_size,
#             nhead=num_heads,
#             dim_feedforward=feedforward_size,
#             dropout=dropout,
#             activation=activation,
#             batch_first=batch_first,
#             norm_first=norm_first,
#         )
#         norm = ReeoLayerNorm(hidden_size, bias=bias)
#         self.transformer = nn.TransformerEncoder(
#             layer, num_layers=num_layers, norm=norm
#         )
#         self._replace(self.transformer)
#         self.loc = nn.Linear(hidden_size, motion_feature_size * step, bias)
#         self.log_scale = nn.Linear(hidden_size, motion_feature_size * step, bias)
#         self.register_buffer(
#             "target_mask",
#             torch.triu(
#                 torch.full((sequence_length, sequence_length), float("-inf")),
#                 diagonal=1,
#             ),
#         )

#     def forward(
#         self,
#         motions: Tensor,
#         audios: Tensor,
#         src_pad_masks: Tensor = None,
#     ) -> Tuple[Tensor, Tensor]:
#         scale = math.sqrt(self.hparams.hidden_size)
#         inputs = torch.cat((motions, audios), dim=-1)
#         inputs = self.linear_emdedding(inputs) * scale
#         inputs = self.pos_embedding(inputs)
#         outputs = self.transformer(
#             inputs,
#             mask=self.target_mask,
#             src_key_padding_mask=src_pad_masks,
#             is_causal=True,
#         )
#         locs = self.loc(outputs)
#         log_scales = self.log_scale(outputs)
#         return locs.real, log_scales.real

#     def _shared_step(self, batch, phase: str) -> Dict[str, Tensor]:
#         motions, audios = batch
#         step = self.hparams.step
#         if torch.rand(1).item() < self.hparams.zero_initial_prob:
#             motions = motions[step:]
#             audios = audios[step:]
#             targets = _torch_stack_targets(motions, step)
#             if torch.rand(1).item() < self.hparams.noise_prob:
#                 motions += (
#                     torch.randn_like(
#                         motions, dtype=motions.dtype, device=motions.device
#                     )
#                     * self.hparams.noise_scale
#                 )
#                 motions = F.pad(motions[:-step], (0, 0, 0, 0, step, 0))
#         else:
#             targets = _torch_stack_targets_V2(motions, step)
#             motions = motions[:-step]
#             audios = audios[step:]
#             if torch.rand(1).item() < self.hparams.noise_prob:
#                 motions += (
#                     torch.randn_like(
#                         motions, dtype=motions.dtype, device=motions.device
#                     )
#                     * self.hparams.noise_scale
#                 )
#         locs, log_scales = self.forward(motions, audios)
#         loss = -Normal(locs, log_scales.exp()).log_prob(targets).mean()
#         self.log(
#             f"{phase}_loss",
#             loss,
#             on_step=False,
#             on_epoch=True,
#             prog_bar=True,
#             logger=True,
#         )
#         return loss

#     def _generate(
#         self,
#         motions: Tensor,
#         audios: Tensor,
#         loc_coef: float = 1.1,
#         scale_coef: float = 1.0,
#         mode: str = "inference",
#     ) -> Tensor:
#         step = self.hparams.step
#         feat_size = self.hparams.motion_feature_size
#         for i in range(step, len(audios) + step, step):
#             locs, log_scales = self(motions[:-step], audios)
#             dists = Normal(locs[i - 1], log_scales[i - 1].exp() * scale_coef)
#             if mode == "inference":
#                 predictions = torch.cat(
#                     (
#                         locs[i - 1].reshape(step, -1, feat_size)[:, :, :3] / loc_coef,
#                         dists.sample().reshape(step, -1, feat_size)[:, :, 3:],
#                     ),
#                     dim=-1,
#                 )
#             else:
#                 predictions = dists.sample().reshape(step, -1, feat_size)
#             motions[i : i + step] = predictions
#         return motions

#     def generate(
#         self,
#         audios: Tensor,
#         loc_coef: float = 1.1,
#         scale_coef: float = 1.0,
#         disable_bar: bool = False,
#     ) -> Tensor:
#         step = self.hparams.step
#         sequence_length = self.hparams.sequence_length
#         feat_size = self.hparams.motion_feature_size

#         pad = sequence_length - (len(audios) % sequence_length)

#         audios = F.pad(audios, (0, 0, 0, 0, 0, pad))

#         motions = torch.zeros(
#             (step + len(audios), audios.shape[1], feat_size),
#             device=audios.device,
#             dtype=audios.dtype,
#         )

#         for i in tqdm(
#             range(sequence_length, len(audios) + 1, sequence_length),
#             disable=disable_bar,
#         ):
#             motions[i - sequence_length : i + step] = self._generate(
#                 motions[i - sequence_length : i + step],
#                 audios[i - sequence_length : i],
#                 loc_coef,
#                 scale_coef,
#                 "inference",
#             )
#         return motions[step:-pad]

#     def log_prob(self, motions: Tensor, audios: Tensor) -> Tensor:
#         """
#         Mathematically, the log joint probability of the whole motion sequence
#         should be the sum of log probability of each features and time step
#         (probability chain rule). However, the sum easily becomes very large
#         and make the exponential infinite.
#         To avoid that, the mean is computed as a scaled down of the log joint
#         probability of the motion sequence.
#         """
#         locs, log_scales = self.forward(
#             F.pad(motions[: -self.hparams.step], (0, 0, 0, 0, self.hparams.step, 0)),
#             audios,
#         )
#         # Take only the immediate next motion to compute log probability:
#         log_probs = (
#             Normal(locs, log_scales.exp())
#             .log_prob(
#                 _torch_stack_targets(
#                     torch.cat((motions, audios), dim=-1), self.hparams.step
#                 )
#             )
#             .mean(dim=(0, 2))
#         )
#         return log_probs
