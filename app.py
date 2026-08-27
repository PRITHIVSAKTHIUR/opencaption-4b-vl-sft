import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import spaces
import torch
import gradio as gr
from gradio import Server
from fastapi.responses import HTMLResponse
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TextIteratorStreamer,
)
from qwen_vl_utils import process_vision_info
from threading import Thread
import base64
from io import BytesIO
from PIL import Image
import ncii_vision_guard

MODEL_ID = "prithivMLmods/OpenCaption-4B-VL-SFT-v1.0"
NCII_BLOCK_MESSAGE = "NCII content detected. Request blocked by safety guard."

SYSTEM_PROMPT = """You are a detailed image captioning assistant.

Structure every caption as follows:

1. Open with one sentence naming the shot type (e.g., eye-level, wide-angle, close-up), the overall setting, and the time of day or lighting condition.

2. Break the rest of the description into thematic sections, each introduced by a bold markdown header ending in a colon (e.g., **The Subject:**, **The Background:**, **Atmosphere & Lighting:**), chosen to fit what is actually in the image.

3. Within each section, use bullet points to list specific, concrete details, including positions, colors, textures, materials, actions, spatial relationships, and any legible text or fine-grained visual elements. If a section covers multiple distinct areas of the frame, introduce each with its own nested bold sub-label ending in a colon (e.g., **Foreground Right:**, **Background:**) before its bullet points.

4. Close with a short, unheaded paragraph beginning with "In summary," that ties the scene together and conveys its overall mood or narrative.

Use precise, sensory, and fluent language throughout. Describe only what is visible in the image, avoid speculation or unsupported inferences, and do not use emojis."""

DEFAULT_PROMPT = "Provide a detailed caption for this image with fine-grained visual details."

print("Loading main captioning processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Loading main captioning model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
).eval()
print("Model loaded.")


def b64_to_pil(b64_str):
    if not b64_str:
        return None
    try:
        if b64_str.startswith("data:image"):
            _, data = b64_str.split(",", 1)
        else:
            data = b64_str
        return Image.open(BytesIO(base64.b64decode(data))).convert("RGB")
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


def make_thumb_b64(path, max_dim=200):
    if not os.path.exists(path):
        return ""
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=60)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception:
        return ""


def encode_full_image(path):
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            data = f.read()
        ext = path.rsplit(".", 1)[-1].lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception:
        return ""


EXAMPLES_CONFIG = [
    {"image": "examples/cafe_interior.jpg"},
    {"image": "examples/dog.jpg"},
    {"image": "examples/cherry_blossom.jpg"},
    {"image": "examples/hot_air_balloon.jpg"},
    {"image": "examples/sushi_platter.jpg"},
]

CLIENT_CONFIG = {
    "examples": [
        {
            "idx": i,
            "path": ex["image"],
            "thumb": make_thumb_b64(ex["image"])
        } for i, ex in enumerate(EXAMPLES_CONFIG)
    ]
}

app = Server(title="OpenCaption-Terminal")


@app.api(name="check_safety")
@spaces.GPU(duration=30, size="xlarge")
def check_safety(image_b64: str) -> dict:
    """Pure GPU safety check. Runs BEFORE the captioning generation."""
    try:
        res = ncii_vision_guard.check_image_safety(image_b64, device="cuda")
        is_nsfw = res.get("nsfw") == 1 or res.get("safe") == 0
        if is_nsfw:
            return {"status": "blocked", "message": NCII_BLOCK_MESSAGE}
        return {"status": "ok"}
    except Exception as e:
        print(f"Safety check error: {e}")
        return {"status": "ok", "warning": "Safety check failed, proceeding."}


@app.api(name="generate")
@spaces.GPU(duration=60, size="xlarge")
def generate(
    image_b64: str,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    """Generates a dense, structured image caption and streams it back."""
    pil_image = b64_to_pil(image_b64)
    if pil_image is None:
        yield "[ERROR] No valid image provided."
        return

    # Defense-in-depth: Re-check inside the generation worker
    res = ncii_vision_guard.check_image_safety(image_b64, device="cuda")
    is_nsfw = res.get("nsfw") == 1 or res.get("safe") == 0
    if is_nsfw:
        yield f"[BLOCKED] {NCII_BLOCK_MESSAGE}"
        return

    if next(model.parameters()).device.type != "cuda":
        model.to("cuda")

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": user_prompt or DEFAULT_PROMPT},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": True,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
    }

    generation_error = {"error": None}

    def _run_generation():
        try:
            model.generate(**generation_kwargs)
        except Exception as e:
            generation_error["error"] = e
            try:
                streamer.end()
            except Exception:
                pass

    thread = Thread(target=_run_generation, daemon=True)
    thread.start()

    buffer = ""
    for new_text in streamer:
        buffer += new_text
        yield buffer

    thread.join(timeout=2.0)

    if generation_error["error"] is not None:
        if buffer.strip():
            yield buffer + f"\n\n[ERROR] {generation_error['error']}"
        else:
            yield f"[ERROR] Inference failed: {generation_error['error']}"
        return

    if not buffer.strip():
        yield "[ERROR] No output was generated."
        return


@app.api(name="load_example", queue=False)
def load_example(idx: float) -> dict:
    """Returns full base64 image for a given example index."""
    try:
        i = int(idx)
    except (ValueError, TypeError):
        i = -1
    if i < 0 or i >= len(EXAMPLES_CONFIG):
        return {"image": "", "name": "", "status": "error"}
    
    path = EXAMPLES_CONFIG[i]["image"]
    b64 = encode_full_image(path)
    return {"image": b64, "name": os.path.basename(path), "status": "ok"}


@app.get("/api/config")
def client_config():
    return CLIENT_CONFIG


@app.get("/", response_class=HTMLResponse)
async def homepage():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    app.launch(show_error=True, mcp_server=True)