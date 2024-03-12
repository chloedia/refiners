import argparse
from pathlib import Path

import torch
from torch import nn

import refiners.fluxion.layers as fl
from refiners.fluxion.model_converter import ModelConverter
from refiners.fluxion.utils import save_to_safetensors
from refiners.foundationals.grounding_DINO.backbone.swin_transformer import SwinTransformerH
from refiners.foundationals.GroundingDINO.groundingdino.util.inference import (
    annotate,  # type: ignore
    load_image,  # type: ignore
    load_model,  # type: ignore
    predict,  # type: ignore
)

x = torch.rand(2, 3, 1024, 1024)

target = load_model(  # type: ignore
    "src/refiners/foundationals/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "src/refiners/foundationals/GroundingDINO/weights/groundingdino_swint_ogc.pth",
).backbone[0]
print("Model Target loaded :)")

source = SwinTransformerH(pretrain_img_size=1024)
print("Model Source loaded :)")

converter = ModelConverter(source_model=source, target_model=target, verbose=True)#type:ignore

converter.run(
        source_args=(x,),
        target_args=(x,))

#y_target = model_target.forward_raw(x)  # type:ignore #note : change x for a nested tensor
print("target ok")

# y_source = model_source(x)

# print(y_target.shape)  # type:ignore
# print(y_source.shape)
