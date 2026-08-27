import torch
from transformers import AutoProcessor
from PIL import Image
import io
import base64
import json
import re

GUARD_MODEL_ID = "prithivMLmods/ImageShield-MMCF-0.8B"

# Pre-load the model
print(f"Loading NCII Vision Guard model: {GUARD_MODEL_ID}")
try:
    from transformers import Qwen3_5ForConditionalGeneration as GuardModelClass
except ImportError:
    try:
        from transformers import AutoModelForVision2Seq as GuardModelClass
    except ImportError:
        from transformers import AutoModelForImageTextToText as GuardModelClass

_guard_processor = AutoProcessor.from_pretrained(GUARD_MODEL_ID)
_guard_model = GuardModelClass.from_pretrained(
    GUARD_MODEL_ID,
    torch_dtype=torch.bfloat16,
)
print("NCII Vision Guard model loaded successfully.")


def check_image_safety(image_b64: str, device="cpu") -> dict:
    """Runs the ImageShield-MMCF model to check for NSFW/NCII content.
    
    Returns a dict like: {"nsfw": 1, "safe": 0, "reason": "..."}
    """
    # Move model to requested device (e.g., 'cuda')
    _guard_model.to(device)
    
    # Decode base64 image
    if image_b64.startswith("data:image"):
        _, data = image_b64.split(",", 1)
    else:
        data = image_b64
        
    pil_image = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": "check the image safe or not"}
            ]
        }
    ]
    
    text = _guard_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _guard_processor(
        text=[text], images=[pil_image], return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        output_ids = _guard_model.generate(**inputs, max_new_tokens=1024)
        
    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = _guard_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    # Parse JSON from response
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            return result
        else:
            return {"nsfw": 0, "safe": 1, "reason": "No JSON found, assumed safe."}
    except Exception as e:
        # Fallback string matching if JSON parsing fails
        if '"nsfw": 1' in response or '"safe": 0' in response:
            return {"nsfw": 1, "safe": 0, "reason": response}
        return {"nsfw": 0, "safe": 1, "reason": "Parsing failed, assumed safe."}