import safetensors.torch
import torch

from refiners.fluxion.model_converter import ModelConverter
from refiners.foundationals.layer_diffuse.models import UNet1024Refiners
from refiners.foundationals.source_layerdiffuse.lib_layerdiffusion.models import (
    UNet1024,
)


def load_torch_file(ckpt, safe_load=False, device=None):  # type: ignore
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors"):  # type: ignore
        sd = safetensors.torch.load_file(ckpt, device=device.type)  # type: ignore
    else:
        if safe_load:
            if not "weights_only" in torch.load.__code__.co_varnames:  # type: ignore
                print("Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely.")
                safe_load = False
        if safe_load:
            pl_sd = torch.load(ckpt, map_location=device, weights_only=True)  # type: ignore
        else:
            pl_sd = torch.load(ckpt, map_location=device, pickle_module=ldm_patched.modules.checkpoint_pickle)  # type: ignore
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    return sd


if __name__ == "__main__":
    target_model = UNet1024Refiners(out_channels=4)
    print("Source Model Loaded")
    source_model = UNet1024(out_channels=4)
    print("Target Model Loaded")
    source_state_dict = load_torch_file("./vae_transparent_decoder.safetensors")
    source_model.load_state_dict(
        source_state_dict  # type: ignore
    )
    source_model.eval()

    x = torch.randn(1, 3, 1024, 1024)
    latent = torch.randn(1, 4, 128, 128)

    # y_source = source_model(x)

    # print("Y_Target computed")
    target_model.set_context("unet1024", {"latent": latent})
    # y_source = source_model(x, latent)
    # print("Y_Source computed")

    # type:ignore

    # print(y_source.shape)
    # print(y_target.shape)  # type:ignore

    converter = ModelConverter(
        source_model=source_model, target_model=target_model, skip_output_check=False, verbose=True, threshold=10
    )
    converter.run(source_args=(x, latent), target_args=(x,))

    y_target = target_model(x)

    print("Y_Target computed")
    y_source = source_model(x, latent)
    print("Y_Source computed")
