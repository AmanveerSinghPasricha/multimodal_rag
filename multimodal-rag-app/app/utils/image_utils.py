import base64
import mimetypes


SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


def encode_image(path: str) -> dict:
    """
    Encode a local image file to a base64 data-URL dict
    compatible with the Groq / OpenAI multimodal message format.
    """
    mime, _ = mimetypes.guess_type(path)
    if mime not in SUPPORTED_IMAGE_TYPES:
        mime = "image/jpeg"  # safe fallback

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }