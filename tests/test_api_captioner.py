
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
from PIL import Image

# Add the project root to sys.path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation_harness import image_utils

class TestApiCaptioner(unittest.TestCase):
    @patch("openai.OpenAI")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_key", "OPENAI_BASE_URL": "test_url"})
    def test_get_captioning_fn_api(self, mock_openai):
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="A test caption"))]
        mock_client.chat.completions.create.return_value = mock_response

        # Get the function
        device = "cpu"
        dtype = "float32"
        model_name = "qwen3-vl-plus"
        caption_fn = image_utils.get_captioning_fn(device, dtype, model_name)

        # Create dummy image
        image = Image.new("RGB", (100, 100), color="red")
        
        # Test 1: VQA (no prompt provided, uses default)
        captions = caption_fn([image])
        self.assertEqual(captions, ["A test caption"])
        
        # Verify API call
        mock_client.chat.completions.create.assert_called()
        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args.kwargs["model"], model_name)
        self.assertEqual(len(call_args.kwargs["messages"]), 1)
        self.assertEqual(call_args.kwargs["messages"][0]["role"], "user")
        
        # Test 2: Custom Prompt
        custom_prompt = ["What color is this?"]
        captions = caption_fn([image], prompt=custom_prompt)
        self.assertEqual(captions, ["A test caption"])
        
        # Verify API call with custom prompt
        call_args = mock_client.chat.completions.create.call_args
        content = call_args.kwargs["messages"][0]["content"]
        self.assertEqual(content[0]["text"], "What color is this?")

if __name__ == "__main__":
    unittest.main()
