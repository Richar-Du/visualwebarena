"""
Quick diagnostic: test LLM API connectivity.

Usage (on the remote machine):
    python test_llm_api.py
"""
import os
import sys
import json

def main():
    print("=" * 60)
    print("  LLM API Connectivity Diagnostic")
    print("=" * 60)

    # 1. Check environment variables
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")

    print(f"\n[1] Environment Variables:")
    print(f"  OPENAI_API_KEY:  {'SET (' + api_key[:8] + '...)' if api_key else 'NOT SET ❌'}")
    print(f"  OPENAI_BASE_URL: {base_url if base_url else 'NOT SET (defaults to https://api.openai.com/v1)'}")

    if not api_key:
        print("\n❌ OPENAI_API_KEY is not set. Cannot proceed.")
        sys.exit(1)

    # 2. Load config to get model name
    config_file = "config_navi_bench.json"
    model = "gpt-4"
    if os.path.exists(config_file):
        with open(config_file) as f:
            config = json.load(f)
        model = config.get("model", {}).get("model", "gpt-4")
    print(f"\n[2] Model from config: {model}")

    # 3. Test API call
    print(f"\n[3] Testing API call...")
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url=base_url if base_url else None,
        )

        print(f"  Client base_url: {client.base_url}")
        print(f"  Sending test request to model '{model}'...")

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say exactly: API_OK"}],
            max_tokens=10,
            temperature=0,
        )

        content = response.choices[0].message.content
        print(f"\n  ✅ SUCCESS! Response: {content}")
        print(f"  Model used: {response.model}")
        print(f"  Usage: {response.usage}")

    except Exception as e:
        err_type = type(e).__name__
        print(f"\n  ❌ FAILED: {err_type}")
        print(f"  Error details: {e}")

        # Provide specific diagnosis
        print(f"\n[4] Diagnosis:")
        err_str = str(e).lower()
        if "authentication" in err_str or "api key" in err_str or "401" in err_str:
            print("  → API key is invalid or not accepted by the server")
            print("  → Check: Is the key correct? Does it match the base_url provider?")
        elif "not found" in err_str or "does not exist" in err_str or "404" in err_str:
            print(f"  → Model '{model}' not found on this API endpoint")
            print(f"  → Check: Does the provider at {base_url or 'api.openai.com'} support '{model}'?")
        elif "rate limit" in err_str or "429" in err_str:
            print("  → Rate limited. The API key may have exhausted its quota.")
        elif "connection" in err_str or "timeout" in err_str:
            print(f"  → Cannot connect to {base_url or 'https://api.openai.com/v1'}")
            print("  → Check: Is there a firewall? Is the URL correct?")
        elif "bad request" in err_str or "400" in err_str:
            print(f"  → Bad request. The model '{model}' may not support the request format.")
            print(f"  → Check: Is the model name exactly correct for this provider?")
        elif "internal server error" in err_str or "500" in err_str:
            print("  → Server-side error. The API provider may be experiencing issues.")
        else:
            print(f"  → Unrecognized error. Full error: {e}")

    # 4. Test multimodal (if model supports it)
    print(f"\n[5] Testing multimodal capability...")
    try:
        from openai import OpenAI
        import base64

        client = OpenAI(
            api_key=api_key,
            base_url=base_url if base_url else None,
        )

        # Create a tiny 1x1 red pixel PNG
        import struct, zlib
        def create_tiny_png():
            raw = b'\x00\xff\x00\x00'  # filter byte + RGB
            compressed = zlib.compress(raw)
            ihdr = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
            def chunk(ctype, data):
                c = ctype + data
                return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
            return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + chunk(b'IDAT', compressed) + chunk(b'IEND', b'')

        png_data = create_tiny_png()
        b64_image = base64.b64encode(png_data).decode()

        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this pixel? Reply with just the color name."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                ]
            }],
            max_tokens=10,
            temperature=0,
        )
        content = response.choices[0].message.content
        print(f"  ✅ Multimodal works! Response: {content}")

    except Exception as e:
        err_type = type(e).__name__
        print(f"  ⚠️  Multimodal failed: {err_type}: {e}")
        print(f"  This may be OK if the model doesn't support vision.")
        print(f"  The agent will fall back to text-only mode.")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
