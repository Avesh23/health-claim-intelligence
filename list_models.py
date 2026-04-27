import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

def list_gemini_models():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env file.")
        return

    client = genai.Client(api_key=api_key)
    print("Available Models:")
    try:
        for model in client.models.list():
            print(f"- {model.name} (Supports: {model.supported_actions})")
    except Exception as e:
        print(f"Failed to list models: {e}")

if __name__ == "__main__":
    list_gemini_models()
