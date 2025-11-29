import sys
import os
import time
from pathlib import Path
from PIL import Image
import numpy as np

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Set default DATASET env var if not present
if "DATASET" not in os.environ:
    os.environ["DATASET"] = "visualwebarena"

from playwright.sync_api import sync_playwright, ViewportSize
from browser_env.processors import ImageObservationProcessor

def main():
    with sync_playwright() as p:
        browser = None
        try:
            # Try to connect to an existing browser instance on port 9222
            print("Attempting to connect to existing browser on port 9222...")
            browser = p.chromium.connect_over_cdp("https://trains.ctrip.com/")
            context = browser.contexts[0]
            print("Connected to existing browser.")
        except Exception as e:
            print(f"Could not connect to existing browser: {e}")
            print("Launching new browser instance...")
            # Launch a new browser instance if connection fails
            browser = p.chromium.launch(headless=False, args=["--remote-debugging-port=9222"])
            context = browser.new_context(viewport={"width": 1280, "height": 2048})
            print("Launched new browser.")

        # Ensure there is at least one page
        if not context.pages:
            page = context.new_page()
        else:
            page = context.pages[0]

        # Initialize the SOM processor
        viewport_size = {"width": 1280, "height": 2048}
        processor = ImageObservationProcessor(
            observation_type="image_som",
            viewport_size=viewport_size
        )

        print("\n--- SOM Parsing Test Script ---")
        print("Press 'Enter' to parse the current page.")
        print("Type 'q' and press 'Enter' to quit.")

        while True:
            user_input = input("\nCommand: ")
            if user_input.lower() == 'q':
                break

            try:
                # Bring page to front to ensure it's active
                page.bring_to_front()
                
                # Process the page
                print("Processing page...")
                start_time = time.time()
                
                # The processor returns a numpy array (image) and a content string (text)
                screenshot_som, content_str = processor.process(page)
                
                end_time = time.time()
                print(f"Processing took {end_time - start_time:.4f} seconds.")

                # Save the SOM image
                output_image_path = "som_result.png"
                Image.fromarray(screenshot_som).save(output_image_path)
                print(f"SOM image saved to: {output_image_path}")

                # Print the text representation
                print("\n--- SOM Text Representation ---")
                print(content_str)
                print("-------------------------------")

            except Exception as e:
                print(f"Error during processing: {e}")
                import traceback
                traceback.print_exc()

        if browser:
            browser.close()

if __name__ == "__main__":
    main()
