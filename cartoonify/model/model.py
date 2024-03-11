import cv2
import torch
import random
import numpy as np
from io import BytesIO
import base64

import PIL
from PIL import Image

import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from huggingface_hub import hf_hub_download

from insightface.app import FaceAnalysis

from pipeline_stable_diffusion_xl_instantid_full import (
    StableDiffusionXLInstantIDPipeline,
    draw_kps,
)

from controlnet_aux import OpenposeDetector

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

import torch.nn.functional as F
from torchvision.transforms import Compose

MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if str(device).__contains__("cuda") else torch.float32
scheduler = "EulerDiscreteScheduler"
guidance_scale = 6.5
identitynet_strength_ratio = 0.8
adapter_strength_ratio = 0.8
num_steps = 30


class Model:
    def __init__(self, **kwargs):
        # Uncomment the following to get access
        # to various parts of the Truss config.

        # self._data_dir = kwargs["data_dir"]
        # self._config = kwargs["config"]
        # self._secrets = kwargs["secrets"]
        self._model = None

    def load(self):
        # Load model here and assign to self._model.
        # hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/config.json", local_dir="./checkpoints")
        # hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/diffusion_pytorch_model.safetensors", local_dir="./checkpoints")
        hf_hub_download(
            repo_id="InstantX/InstantID",
            filename="ip-adapter.bin",
            local_dir="./checkpoints",
        )

        app = FaceAnalysis(
            name="antelopev2", root="./", providers=["CPUExecutionProvider"]
        )
        app.prepare(ctx_id=0, det_size=(640, 640))

        face_adapter = f"./checkpoints/ip-adapter.bin"
        controlnet_path = f"./checkpoints/ControlNetModel"

        controlnet_identitynet = ControlNetModel.from_pretrained(
            controlnet_path, torch_dtype=dtype
        )

        pretrained_model_name_or_path = "wangqixun/YamerMIX_v8"
        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=dtype,
            safety_checker=None,
            feature_extractor=None,
            controlnet=[controlnet_identitynet],
        ).to(device)

        pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(
            pipe.scheduler.config
        )

        pipe.cuda()
        pipe.load_ip_adapter_instantid(face_adapter)
        pipe.image_proj_model.to("cuda")
        pipe.unet.to("cuda")

        self._model = pipe
        self._app = app
        self._controlnet_identitynet = controlnet_identitynet

    def predict(self, model_input):
        pipe = self._model
        app = self._app
        controlnet_identitynet = self._controlnet_identitynet
        prompts = model_input["prompts"]  # max 10
        face_image_paths = model_input["face_image_paths"]  # max 2
        pose_image_paths = model_input["pose_image_paths"]  # max 2

        image_results = []
        negative_prompt = """(lowres, low quality, worst quality:1.2), (text:1.2), watermark, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, glitch,deformed, mutated, cross-eyed, ugly, disfigured"""
        for face_image_path in face_image_paths:
            for pose_image_path in pose_image_paths:
                for prompt in prompts:
                    image = generate_image(
                        face_image_path,
                        pose_image_path,
                        prompt,
                        negative_prompt,
                        pipe,
                        app,
                        controlnet_identitynet,
                    )
                    image_results.append(
                        {
                            "data": image_to_base64(image),
                            "format": "png",
                        }
                    )

        return image_results


def image_to_base64(image: Image.Image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def resize_img(
    input_image,
    max_side=1280,
    min_side=1024,
    size=None,
    pad_to_max_side=False,
    mode=PIL.Image.BILINEAR,
    base_pixel_number=64,
):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y : offset_y + h_resize_new, offset_x : offset_x + w_resize_new] = (
            np.array(input_image)
        )
        input_image = Image.fromarray(res)
    return input_image


def generate_image(
    face_image_path,
    pose_image_path,
    prompt,
    negative_prompt,
    pipe,
    app,
    controlnet_identitynet,
):
    scheduler = getattr(diffusers, scheduler)
    pipe.scheduler = scheduler.from_config(pipe.scheduler.config, {})

    face_image = load_image(face_image_path)
    face_image = resize_img(face_image, max_side=1024)
    face_image_cv2 = convert_from_image_to_cv2(face_image)
    height, width, _ = face_image_cv2.shape

    # Extract face features
    face_info = app.get(face_image_cv2)

    if len(face_info) == 0:
        return None

    face_info = sorted(
        face_info,
        key=lambda x: (x["bbox"][2] - x["bbox"][0]) * x["bbox"][3] - x["bbox"][1],
    )[
        -1
    ]  # only use the maximum face
    face_emb = face_info["embedding"]
    face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_info["kps"])

    if pose_image_path is not None:
        pose_image = load_image(pose_image_path)
        pose_image = resize_img(pose_image, max_side=1024)

        pose_image_cv2 = convert_from_image_to_cv2(pose_image)
        face_info = app.get(pose_image_cv2)

        if len(face_info) == 0:
            return None

        face_info = face_info[-1]
        face_kps = draw_kps(pose_image, face_info["kps"])
        width, height = face_kps.size

    control_mask = np.zeros([height, width, 3])
    x1, y1, x2, y2 = face_info["bbox"]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    control_mask[y1:y2, x1:x2] = 255
    control_mask = Image.fromarray(control_mask.astype(np.uint8))

    pipe.controlnet = controlnet_identitynet
    control_scales = float(identitynet_strength_ratio)
    control_images = face_kps

    seed = randomize_seed_fn(0, True)
    generator = torch.Generator(device=device).manual_seed(seed)

    pipe.set_ip_adapter_scale(adapter_strength_ratio)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_embeds=face_emb,
        image=control_images,
        control_mask=control_mask,
        controlnet_conditioning_scale=control_scales,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
    ).images[0]
    return image
