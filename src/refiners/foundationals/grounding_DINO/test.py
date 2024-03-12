import torch

from refiners.foundationals.grounding_DINO.backbone.swin_transformer import SwinTransformerH
from refiners.foundationals.GroundingDINO.groundingdino.util.inference import (
    annotate,  # type: ignore
    load_image,  # type: ignore
    load_model,  # type: ignore
    predict,  # type: ignore
)

x = torch.rand(2, 3, 1024, 1024)

# model_target = load_model(  # type: ignore
#     "src/refiners/foundationals/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
#     "src/refiners/foundationals/GroundingDINO/weights/groundingdino_swint_ogc.pth",
# ).backbone[0]
# print("Model Target loaded :)")

# y_target = model_target.forward_raw(x)  # type:ignore #note : change x for a nested tensor
# print("target ok")

model_source = SwinTransformerH(pretrain_img_size=1024)
print("Model Source loaded :)")

y_source = model_source(x)

print(y_target.shape)  # type:ignore
print(y_source.shape)
