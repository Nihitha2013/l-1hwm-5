import requests
from PIL import Image
from io import BytesIO
from config import HF_API_KEY  

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3-medium-diffusers"


def generate_image_from_text(prompt: str) -> Image.Image:
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": "ugly, blurry, distorted, low quality",
            "guidance_scale": 7.5
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "image" in content_type:
            return Image.open(BytesIO(response.content))
        else:
            try:
                error = response.json()
                raise Exception(f"Hugging Face API error: {error}")
            except ValueError:
                raise Exception("Unexpected response format (not an image or JSON).")

    except requests.exceptions.RequestException as e:
        raise Exception(f"Request failed: {e}")


def main():
    """
    Runs a CLI interface for generating images from text prompts.
    """
    print("🎨 Welcome to the Text-to-Image Generator!")
    print("Type 'exit' to quit the program.\n")

    while True:
        prompt = input("Enter a description for the image you want to generate:\n").strip()
        if prompt.lower() == "exit":
            print("Goodbye!")
            break

        print("\nGenerating image... Please wait.\n")

        try:
            image = generate_image_from_text(prompt)
            image.show()

            save_option = input("Do you want to save this image? (yes/no): ").strip().lower()
            if save_option == "yes":
                file_name = input("Enter a name for the image file (without extension): ").strip() or "generated_image"
                file_name = "".join(c for c in file_name if c.isalnum() or c in ("_", "-")).rstrip()
                file_path = f"{file_name}.png"
                image.save(file_path)
                print(f" Image saved as {file_path}\n")

        except Exception as e:
            print(f"❌An error occurred: {e}\n")

        print("-" * 80 + "\n")


if __name__ == "__main__":
    main()
