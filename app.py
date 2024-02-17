from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from time import time

import cv2
import gradio as gr
import numpy as np
import torch
from cv2 import imencode
from diffusers import AutoencoderTiny, LCMScheduler
from diffusers.models import ControlNetModel
from insightface.app import FaceAnalysis
from PIL import Image

from download_models import download_instant_id_sdxl_models
from pipeline_stable_diffusion_xl_instantid import (
    StableDiffusionXLInstantIDPipeline,
    draw_kps,
)


# https://github.com/gradio-app/gradio/issues/2635#issuecomment-1423531319
def encode_pil_to_base64_new(pil_image):
    print("-> encode_pil_to_base64_new")
    image_arr = np.asarray(pil_image)[:, :, ::-1]
    _, byte_data = imencode(".png", image_arr)
    base64_data = b64encode(byte_data)
    base64_string_opencv = base64_data.decode("utf-8")
    return "data:image/png;base64," + base64_string_opencv


# monkey patching encode pil
gr.processing_utils.encode_pil_to_base64 = encode_pil_to_base64_new

face_adapter = f"./models/sdxl/ip-adapter.bin"
controlnet_path = f"./models/sdxl/ControlNetModel"
base_model = "stabilityai/sdxl-turbo"
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

# pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
pipe.vae = AutoencoderTiny.from_pretrained(
    "madebyollin/taesdxl",
    torch_dtype=torch_dtype,
)
pipe.text_encoder = pipe.text_encoder.to(memory_format=torch.channels_last)
pipe.scheduler = LCMScheduler.from_config(
    pipe.scheduler.config,
    beta_start=0.001,
    beta_end=0.012,
)

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
):
    start = time()
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
    face_kps.save("kps.jpg")
    print(time() - start)

    # generate image
    pipe.set_ip_adapter_scale(0.8)
    print("start")
    images = pipe(
        prompt,
        image_embeds=face_emb,
        image=face_kps,
        controlnet_conditioning_scale=0.8,
        num_inference_steps=1,
        guidance_scale=0.0,
    ).images
    return images


def process_image(
    face_image,
    prompt,
):
    tick = time()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            generate_image,
            face_image,
            prompt,
        )
        images = future.result()
    elapsed = time() - tick
    print(f"Latency : {elapsed:.2f} seconds")
    return images[0]


def _get_footer_message() -> str:
    version = f"<center><p> {APP_VERSION} "
    footer_msg = version + (
        '  © 2024 <a href="https://github.com/rupeshs">'
        " Rupesh Sreeraman</a></p></center>"
    )
    return footer_msg


def get_web_ui() -> gr.Blocks:
    with gr.Blocks(
        title="InstantID CPU",
    ) as web_ui:
        gr.HTML("<center><H1>InstantID CPU</H1></center>")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Init image",
                    type="pil",
                    height=512,
                )
                with gr.Row():
                    prompt = gr.Textbox(
                        show_label=False,
                        lines=3,
                        placeholder="A photo of a person",
                        container=False,
                    )

                    generate_btn = gr.Button(
                        "Generate",
                        elem_id="generate_button",
                        scale=0,
                    )

                input_params = [
                    input_image,
                    prompt,
                ]

            with gr.Column():
                gallery = gr.Image(label="Generated Images")
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
