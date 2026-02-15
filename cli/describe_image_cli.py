import argparse
import mimetypes
import os
import sys

from google import genai
from google.genai import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# from src.hybrid_search import HybridSearch
def main(raw_args: list[str] | None = None):
    parser = argparse.ArgumentParser(description="describe image CLI")
    _ = parser.add_argument("--image", type=str, required=True, help="The path to an image file")
    _ = parser.add_argument("--query", type=str, required=True, help="A text query to rewrite based on the image")
    args = parser.parse_args(raw_args)

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"
    with open(args.image, "rb") as f:
        image_data = f.read()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    client = genai.Client(api_key=api_key)
    system_prompt = """Given the included image and text query,
    rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary"""
    parts = [
        system_prompt,
        types.Part.from_bytes(data=image_data, mime_type=mime),
        args.query.strip(),
    ]
    response = client.models.generate_content(model="gemini-2.5-flash", contents=parts)
    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")
    # run evaluation logic here


if __name__ == "__main__":
    main()
