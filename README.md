# **[OpenCaption-4B-VL-SFT](https://huggingface.co/spaces/prithivMLmods/opencaption-4b-vl-sft)**

OpenCaption-4B-VL-SFT is an advanced multimodal image captioning terminal interface powered by `prithivMLmods/OpenCaption-4B-VL-SFT-v1.0`. Built upon the `Qwen3VLForConditionalGeneration` architecture, this platform generates highly structured, fine-grained visual descriptions with theme-based sections, spatial bounding references, and concrete sensory details.

The application is deployed as a retro terminal-styled single-page application (SPA) backed by a FastAPI engine (`gradio.Server`). It features streaming text generation via `TextIteratorStreamer`, built-in CLI command handling, inline image previews, and zero-GPU resource allocation hooks.

<img width="1919" height="830" alt="Screenshot 2026-08-21 230426" src="https://github.com/user-attachments/assets/4a392083-0de6-472e-bfae-8568ec691840" />
<img width="1919" height="840" alt="Screenshot 2026-08-21 230445" src="https://github.com/user-attachments/assets/69c865da-2c7d-437b-adda-6414341ec01f" />

### **Key Features**

* **Dense Structured Captioning:** Generates rich captions containing:
1. Opening shot type, overall setting, and lighting conditions.
2. Thematic sections (e.g., `**The Subject:**`, `**The Background:**`, `**Atmosphere & Lighting:**`).
3. Spatial bullet points detailing colors, textures, materials, and spatial relationships.
4. Closing summary synthesis.

* **Token Streaming:** Employs `TextIteratorStreamer` with client-side markdown parsing (`marked.js`) to stream generated tokens in real-time with an active cursor.
* **Retro CLI Terminal Workspace:** A full-featured terminal interface supporting both CLI commands (`upload`, `generate`, `copy`, `clear`, `help`) and interactive quick-action buttons.
* **Aspect-Preserving Preprocessing:** Resizes and crops input images using `qwen_vl_utils.process_vision_info` for optimal token length and memory efficiency.
* **Dynamic Example Browser:** One-click loading of preset images directly into the command line buffer via dedicated API endpoints.

### **Repository Structure**

```text
├── examples/
│   ├── cafe_interior.jpg
│   ├── cherry_blossom.jpg
│   ├── dog.jpg
│   ├── hot_air_balloon.jpg
│   └── sushi_platter.jpg
├── src/
│   └── opencaption_4b_vl_sft/
│       └── __init__.py
├── app.py
├── index.html
├── LICENSE.txt
├── pre-requirements.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock
```

### **Installation and Requirements**

To configure the OpenCaption-4B-VL-SFT workspace locally, ensure your environment meets the following specifications:

* **Python Version:** Minimum Python **3.13** or above is required.
* **PyTorch Version:** `torch==2.11.0` or above is required for best compatibility.
* **CUDA Version:** CUDA **13.0** is recommended, matching the environment running on the live Hugging Face Space.

#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust. It ensures rapid virtual environment synchronization and deterministic dependency management based on `uv.lock`.

**Step 1 — Install `uv`**

* **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Windows:** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

**Step 2 — Clone the repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/opencaption-4b-vl-sft.git
cd opencaption-4b-vl-sft
```

**Step 3 — Initialize the project and install dependencies**

```bash
uv sync
```

**Step 4 — Run the script**

```bash
uv run app.py
```

#### **Standard PIP Implementation**

**1. Update Package Manager**
Upgrade your local package manager:

```bash
pip install pip>=26.1.2
```

**2. Install Core Dependencies**
Install the primary deep learning stack, transformers, and vision-language utilities listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

#### **Core Requirements List (`requirements.txt`)**

```text
--extra-index-url https://download.pytorch.org/whl/cu130
torch==2.11.0
torchvision>=0.28.0
transformers==5.14.1
accelerate>=1.14.0
qwen-vl-utils>=0.0.14
gradio==6.22.0
spaces==0.51.1
pillow==12.3.0
huggingface-hub>=1.27.0
```

### **Usage**

Once the server initializes, open your browser to the local address output in your terminal (typically `http://127.0.0.1:7860/`).

1. **Load Image:**
* Click **[ Upload ]** or type `upload` in the CLI to select a local image.
* Click any example file path listed under `EXAMPLES_DIR:`.

2. **Generate Caption:**
* Click **[ Generate ]** or type `generate` and press Enter.
* The model will stream the fine-grained visual caption directly into the terminal window.

3. **Copy Output:**
* Click **[ Copy ]** or type `copy` to copy the generated markdown to your clipboard.

4. **Clear Session:**
* Click **[ Clear ]** or type `clear` to reset the terminal buffer and start a new session.

### **License and Source**

* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/opencaption-4b-vl-sft/blob/main/LICENSE.txt)
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/opencaption-4b-vl-sft.git](https://github.com/PRITHIVSAKTHIUR/opencaption-4b-vl-sft.git)
* **Hugging Face Live Space:** [https://huggingface.co/spaces/prithivMLmods/opencaption-4b-vl-sft](https://huggingface.co/spaces/prithivMLmods/opencaption-4b-vl-sft)
