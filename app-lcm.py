from concurrent.futures import ThreadPoolExecutor
from time import time

import cv2
import gradio as gr
import numpy as np
import torch
from diffusers import LCMScheduler
from diffusers.models import ControlNetModel
from insightface.app import FaceAnalysis

from download_models import download_instant_id_sdxl_models
from pipeline_stable_diffusion_xl_instantid import (
    StableDiffusionXLInstantIDPipeline,
    draw_kps,
)

face_adapter = f"./models/sdxl/ip-adapter.bin"
controlnet_path = f"./models/sdxl/ControlNetModel"
base_model = "wangqixun/YamerMIX_v8"
APP_VERSION = "0.1.0"
device = "cpu"

download_instant_id_sdxl_models()

app = FaceAnalysis(
    name="antelopev2",
    root="./",
    providers=["CPUExecutionProvider"],
)
app.prepare(ctx_id=0, det_size=(320, 320))
torch_dtype = torch.float32

controlnet = ControlNetModel.from_pretrained(
    controlnet_path,
    torch_dtype=torch_dtype,
)
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    base_model,
    controlnet=controlnet,
    torch_dtype=torch_dtype,
)
pipe.load_ip_adapter_instantid(face_adapter)
pipe.to(device)
pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
pipe.controlnet = pipe.controlnet.to(memory_format=torch.channels_last)
pipe.text_encoder = pipe.text_encoder.to(memory_format=torch.channels_last)
pipe.scheduler = LCMScheduler.from_config(
    pipe.scheduler.config,
    beta_start=0.001,
    beta_end=0.012,
)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
pipe.fuse_lora()
# pipe.scheduler = DEISMultistepScheduler.from_config(
#     pipe.scheduler.config,
# )
pipe.enable_freeu(
    s1=0.6,
    s2=0.4,
    b1=1.1,
    b2=1.2,
)
print(pipe)


def generate_image(
    face_image,
    prompt,
    identitynet_strength_ratio,
    adapter_strength_ratio,
):
    print(f"identitynet_strength_ratio :{identitynet_strength_ratio}")
    print(f"adapter_strength_ratio :{adapter_strength_ratio}")
    if prompt == "":
        prompt = "Photo of a person,high quality"
    face_image = face_image.resize((960, 1024))
    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(
        face_info,
        key=lambda x: (x["bbox"][2] - x["bbox"][0]) * x["bbox"][3] - x["bbox"][1],
    )[
        -1
    ]  # only use the maximum face
    face_emb = face_info["embedding"]
    face_kps = draw_kps(face_image, face_info["kps"])

    # generate image
    pipe.set_ip_adapter_scale(adapter_strength_ratio)
    print("start")
    images = pipe(
        prompt,
        image_embeds=face_emb,
        image=face_kps,
        controlnet_conditioning_scale=identitynet_strength_ratio,
        num_inference_steps=3,
        guidance_scale=0.0,
    ).images
    return images


def process_image(
    face_image,
    prompt,
    identitynet_strength_ratio,
    adapter_strength_ratio,
):
    tick = time()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            generate_image,
            face_image,
            prompt,
            identitynet_strength_ratio,
            adapter_strength_ratio,
        )
        images = future.result()
    elapsed = time() - tick
    print(f"Latency : {elapsed:.2f} seconds")
    return images[0]


def _get_footer_message() -> str:
    version = f"<center><p> v{APP_VERSION}"
    footer_msg = version
    return footer_msg


css = """
#generate_button {
    color: white;
    border-color: #007bff;
    background: #2563eb;

}
"""


def get_web_ui() -> gr.Blocks:
    with gr.Blocks(
        css=css,
        title="InstantID CPU",
    ) as web_ui:
        gr.HTML("<center><H1>InstantID CPU</H1></center>")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Face image",
                    type="pil",
                    height=512,
                )
                with gr.Row():
                    prompt = gr.Textbox(
                        show_label=False,
                        lines=3,
                        placeholder="Oil painting",
                        container=False,
                    )

                    generate_btn = gr.Button(
                        "Generate",
                        elem_id="generate_button",
                        scale=0,
                    )
                identitynet_strength_ratio = gr.Slider(
                    label="IdentityNet strength (for fidelity)",
                    minimum=0,
                    maximum=1.5,
                    step=0.05,
                    value=0.80,
                )
                adapter_strength_ratio = gr.Slider(
                    label="Image adapter strength (for detail)",
                    minimum=0,
                    maximum=1.5,
                    step=0.05,
                    value=0.80,
                )

                input_params = [
                    input_image,
                    prompt,
                    identitynet_strength_ratio,
                    adapter_strength_ratio,
                ]

            with gr.Column():
                gallery = gr.Image(
                    label="Generated Image",
                )
            generate_btn.click(
                fn=process_image,
                inputs=input_params,
                outputs=gallery,
            )

        gr.HTML(_get_footer_message())

    return web_ui


def start_webui(
    share: bool = False,
):
    webui = get_web_ui()
    webui.queue()
    webui.launch(share=share)


start_webui()
