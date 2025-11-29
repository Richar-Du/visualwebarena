from typing import List

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from openai import OpenAI
import base64
import io
import os
from browser_env.utils import pil_to_b64


def get_captioning_fn(
    device, dtype, model_name: str = "qwen3-vl-plus"
) -> callable:   
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL"))
        
        def caption_images(
            images: List[Image.Image],
            prompt: List[str] = None,
            max_new_tokens: int = 300, # Increased default for API
        ) -> List[str]:
            captions = []
            
            # Prepare prompts
            if prompt is None:
                prompts = ["Describe this image in detail."] * len(images)
            else:
                prompts = prompt
                
            assert len(images) == len(prompts), "Number of images and prompts must match"

            for img, p in zip(images, prompts):                
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": p},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": pil_to_b64(img)
                                        },
                                    },
                                ],
                            }
                        ],
                        max_tokens=max_new_tokens,
                    )
                    captions.append(response.choices[0].message.content)
                except Exception as e:
                    print(f"Error calling API: {e}")
                    captions.append("")
            
            return captions

        return caption_images


def get_image_ssim(imageA, imageB):
    # Determine the size to which we should resize
    new_size = max(imageA.size[0], imageB.size[0]), max(
        imageA.size[1], imageB.size[1]
    )

    # Resize images
    imageA = imageA.resize(new_size, Image.LANCZOS)
    imageB = imageB.resize(new_size, Image.LANCZOS)

    # Convert images to grayscale
    grayA = imageA.convert("L")
    grayB = imageB.convert("L")

    # Convert grayscale images to numpy arrays for SSIM computation
    grayA = np.array(grayA)
    grayB = np.array(grayB)

    # Compute the Structural Similarity Index (SSIM) between the two images
    score, _ = ssim(grayA, grayB, full=True)
    return score
