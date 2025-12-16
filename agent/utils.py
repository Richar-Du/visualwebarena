"""Utility functions for the multi-agent system."""

from typing import List


# List of model names/patterns that support multimodal inputs
MULTIMODAL_MODELS = [
    "gpt-4o",
    "gpt-4-vision",
    "gpt-4-turbo",
    "gemini",
    "claude-3",
    "qwen",  # Qwen VL models
]


def is_multimodal_model(model_name: str) -> bool:
    """Check if a model supports multimodal (vision) inputs.
    
    Args:
        model_name: The name of the model to check
        
    Returns:
        True if the model supports multimodal inputs, False otherwise
    """
    model_name_lower = model_name.lower()
    return any(m in model_name_lower for m in MULTIMODAL_MODELS)


def get_multimodal_models() -> List[str]:
    """Get the list of known multimodal model patterns.
    
    Returns:
        List of model name patterns that support multimodal inputs
    """
    return MULTIMODAL_MODELS.copy()

