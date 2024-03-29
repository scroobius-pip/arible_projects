import diffusers

# inputs
# get prompts, get negative prompts, get reference images
prompts = ["Dashing portrait of person in the 1930s, conveying strength and sureness"]
negative_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, deformed, mutated, cross-eyed, ugly, disfigured"
ref_image_urls = [
    "https://i.pinimg.com/564x/b9/58/15/b95815e8fe44c2a2d25d118bfe4a68fe.jpg"
]

# load instantid sdxl pipeline
# https://civitai.com/api/download/models/294470?type=Model&format=SafeTensor&size=pruned&fp=fp16 (realistick stock photo)

# get latent and upscale latent

# send latent to img2img pipeline

# retrieve latent from pipeline

#
