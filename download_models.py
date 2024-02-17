from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download


def download_instant_id_sdxl_models():
    hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ControlNetModel/config.json",
        local_dir="./models/sdxl/",
    )
    hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ControlNetModel/diffusion_pytorch_model.safetensors",
        local_dir="./models/sdxl/",
    )
    hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ip-adapter.bin",
        local_dir="./models/sdxl/",
    )
    snapshot_download(
        repo_id="rupeshs/antelopev2",
        local_dir="./models/antelopev2",
    )
