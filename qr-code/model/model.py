import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
)
import qrcode
import os
import base64
from io import BytesIO

from pyzbar.pyzbar import decode
from PIL import Image

MODEL_ID = "simdi/colorful_qr"
WIDTH = 768
HEIGHT = 768

WEIGHT_PAIRS = [
    (0.25, 0.20),
    (0.25, 0.25),
    (0.35, 0.20),
    (0.35, 0.25),
    (0.45, 0.20),
    (0.45, 0.25),
]


def float_to_pair_index(f: float):
    length = len(WEIGHT_PAIRS)
    # If f is less than length, convert to integer and use directly
    if f < length:
        return int(f)
    # If f is greater or equal to length, assume it's a proportion of the length
    else:
        # Ensuring f is between 0 and 1
        f = max(0.0, min(f, 1.0))
        # Convert the float to an index
        index = int(f * length)
        # Make sure the index is in the valid range
        index = min(index, length - 1)
        return index


def select_weight_pair(f: float):
    return WEIGHT_PAIRS[float_to_pair_index(f)]


def load_models():
    controlnet_tile = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1e_sd15_tile",
        torch_dtype=torch.float16,
    )

    controlnet_brightness = ControlNetModel.from_pretrained(
        "ioclab/control_v1p_sd15_brightness",
        torch_dtype=torch.float16,
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        MODEL_ID,
        controlnet=[
            controlnet_tile,
            controlnet_brightness,
        ],
        torch_dtype=torch.float16,
        cache_dir="cache",
        # local_files_only=True,
    ).to("cuda")

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    return pipe


def resize_for_condition_image(input_image: Image.Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


def generate_qr_code(content: str):
    qrcode_generator = qrcode.QRCode(
        version=1,
        error_correction=qrcode.ERROR_CORRECT_H,
        box_size=10,
        border=2,
    )
    qrcode_generator.clear()
    qrcode_generator.add_data(content)
    qrcode_generator.make(fit=True)
    img = qrcode_generator.make_image(fill_color="black", back_color="white")
    img = resize_for_condition_image(img, 768)
    return img


def is_qr_code_readable(image: Image.Image):
    """
    Check if a given image has a readable QR code.
    Args:
        image (PIL.Image.Image): PIL Image object.
    Returns:
        bool: True if the image contains a readable QR code, False otherwise.
    """

    decoded_objects = decode(image)
    for obj in decoded_objects:
        if obj.type == "QRCODE":
            return True
    return False


def image_to_base64(image: Image.Image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def generate_image_with_conditioning_scale(**inputs):
    prompt = inputs["prompt"]
    pair = inputs["pair"]
    pipe = inputs["pipe"]
    qr_image = inputs["qr_image"]
    generator = inputs["generator"]

    images = pipe(
        prompt=[prompt] * 1,
        negative_prompt=[""] * 1,
        width=WIDTH,
        height=HEIGHT,
        guidance_scale=7.0,
        generator=generator,
        num_inference_steps=25,
        num_images_per_prompt=3,
        controlnet_conditioning_scale=pair,
        image=[qr_image] * 2,
    ).images

    return [{"data": image_to_base64(image), "format": "png"} for image in images]


def generate_image(pipe, inputs):
    prompt = inputs["prompt"]
    content = inputs["content"]
    art_scale = inputs["art_scale"]
    # pipe = inputs["pipe"]
    # torch.backends.cuda.matmul.allow_tf32 = True

    with torch.inference_mode():
        with torch.autocast("cuda"):
            # os.makedirs("images", exist_ok=True)

            qr_image = generate_qr_code(content)
            generator = torch.Generator()
            pair = select_weight_pair(art_scale)
            return generate_image_with_conditioning_scale(
                prompt=prompt,
                pair=pair,
                pipe=pipe,
                qr_image=qr_image,
                generator=generator,
            )


class Model:
    def __init__(self, **kwargs):
        # Uncomment the following to get access
        # to various parts of the Truss config.

        # self._data_dir = kwargs["data_dir"]
        # self._config = kwargs["config"]
        # self._secrets = kwargs["secrets"]
        # self._model = None
        self._model = None

    def load(self):
        # Load model here and assign to self._model.
        self._model = load_models()

    def predict(self, model_input):

        images = generate_image(self._model, model_input)
        return images
