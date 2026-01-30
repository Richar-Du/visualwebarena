"""API-based image captioning and VQA utilities for evaluation.

This module provides OpenAI API-based image captioning functions that can be used
as drop-in replacements for local BLIP-2 models in evaluation.
"""

import base64
import os
from io import BytesIO
from typing import List, Optional, Dict, Any

from PIL import Image
from openai import OpenAI


def get_api_captioning_fn(
    model: str = "gpt-5.1",
    max_tokens: int = 1024,
    temperature: float = 0.3,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> callable:
    """Create an API-based captioning function compatible with PageImageEvaluator.
    
    Args:
        model: The model name to use (e.g., "gpt-5.1", "gpt-4o")
        max_tokens: Maximum tokens for the response
        temperature: Temperature for generation
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        base_url: OpenAI API base URL (defaults to OPENAI_BASE_URL env var)
        
    Returns:
        A callable function that takes images and prompts, returns captions
    """
    # Initialize OpenAI client
    client = OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
    )
    
    def pil_to_base64(img: Image.Image) -> str:
        """Convert PIL Image to base64 data URI."""
        with BytesIO() as buffer:
            img.save(buffer, format="PNG")
            byte_data = buffer.getvalue()
            b64_str = base64.b64encode(byte_data).decode("utf-8")
            return f"data:image/png;base64,{b64_str}"
    
    def caption_images(
        images: List[Image.Image],
        prompt: List[str] = None,
        max_new_tokens: int = 32,
    ) -> List[str]:
        """Generate captions or answers for images using API.
        
        This function is designed to be a drop-in replacement for the BLIP-2
        based captioning function used in PageImageEvaluator.
        
        Args:
            images: List of PIL Images to caption/query
            prompt: List of prompts/questions, one per image. If None, generates captions.
            max_new_tokens: Not used (kept for compatibility), uses model's max_tokens instead
            
        Returns:
            List of captions or answers, one per image
        """
        results = []
        
        for i, img in enumerate(images):
            # Get the prompt for this image
            if prompt is not None and i < len(prompt):
                question = prompt[i]
            else:
                question = "Describe this image in detail."
            
            # Convert image to base64
            img_b64 = pil_to_base64(img)
            
            # Build the message with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": img_b64}
                        },
                        {
                            "type": "text",
                            "text": question
                        }
                    ]
                }
            ]
            
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                answer = response.choices[0].message.content.strip()
                results.append(answer)
            except Exception as e:
                print(f"[API Captioning Error] {e}")
                results.append("")
        
        return results
    
    return caption_images


def get_api_captioning_fn_from_config(config: Dict[str, Any]) -> callable:
    """Create an API-based captioning function from configuration dictionary.
    
    Args:
        config: Configuration dictionary with 'eval' section containing:
            - provider: "openai" (currently only openai is supported)
            - model: Model name (e.g., "gpt-5.1")
            - max_tokens: Maximum tokens for response
            - temperature: Temperature for generation
            
    Returns:
        A callable captioning function
    """
    eval_config = config.get("eval", {})
    
    provider = eval_config.get("provider", "openai")
    if provider != "openai":
        raise ValueError(f"Eval provider '{provider}' not supported. Use 'openai'.")
    
    model = eval_config.get("model", "gpt-5.1")
    max_tokens = eval_config.get("max_tokens", 1024)
    temperature = eval_config.get("temperature", 0.3)
    
    return get_api_captioning_fn(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
