import math

import torch
from torch import Tensor, device as Device, dtype as DType

import refiners.fluxion.layers as fl


class PositionalEmbeddingSine(fl.Residual):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self,
        num_positions: int = 64,
        tempH: int = 10000,
        tempW: int = 10000,
        normalize: bool = False,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.num_positions = num_positions
        self.tempH = tempH
        self.tempW = tempW
        self.normalize = normalize
        self.scale = 2 * math.pi

   
        super().__init__(
                fl.UseContext("backbone_img", "cache"),
                fl.Parameter(
                    num_positions,
                    tempH,
                    tempW,
                    device=device,
                    dtype=dtype,
                ),
                fl.Lambda(self.get_pos),
            )
        

    def get_pos(self, mask: Tensor):
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_tx = torch.arange(self.num_positions, dtype=torch.float32)
        dim_tx = self.tempW ** (2 * (torch.div(dim_tx, 2, rounding_mode="floor")) / self.num_positions)
        pos_x = x_embed[:, :, :, None] / dim_tx

        dim_ty = torch.arange(self.num_positions, dtype=torch.float32)
        dim_ty = self.tempH ** (2 * (torch.div(dim_ty, 2, rounding_mode="floor")) / self.num_positions)
        pos_y = y_embed[:, :, :, None] / dim_ty

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos
