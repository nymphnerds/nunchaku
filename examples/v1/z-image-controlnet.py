import torch
from diffusers import ZImageControlNetModel, ZImageControlNetPipeline
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download

from nunchaku import NunchakuZImageTransformer2DModel
from nunchaku.utils import get_precision, is_turing

if __name__ == "__main__":
    precision = get_precision()
    rank = 128
    dtype = torch.float16 if is_turing() else torch.bfloat16

    controlnet_path = hf_hub_download(
        "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union",
        filename="Z-Image-Turbo-Fun-Controlnet-Union.safetensors",
    )
    controlnet = ZImageControlNetModel.from_single_file(controlnet_path, torch_dtype=dtype)
    transformer = NunchakuZImageTransformer2DModel.from_pretrained(
        f"nunchaku-ai/nunchaku-z-image-turbo/svdq-{precision}_r{rank}-z-image-turbo.safetensors",
        torch_dtype=dtype,
    )

    pipe = ZImageControlNetPipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        transformer=transformer,
        controlnet=controlnet,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    pipe.enable_sequential_cpu_offload()

    control_image = load_image(
        "https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union/resolve/main/asset/pose.jpg"
    )
    prompt = (
        "A crisp fantasy character sprite in a neutral studio pose, full body, readable silhouette, "
        "clean edges, colorful adventuring clothes, game asset concept art"
    )

    image = pipe(
        prompt=prompt,
        control_image=control_image,
        controlnet_conditioning_scale=0.75,
        height=1024,
        width=1024,
        num_inference_steps=8,
        guidance_scale=0.0,
        generator=torch.Generator("cuda").manual_seed(43),
    ).images[0]

    image.save(f"z-image-controlnet-{precision}_r{rank}_{str(dtype)}.png")
