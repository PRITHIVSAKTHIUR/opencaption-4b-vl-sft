# **[OpenCaption-4B-VL-SFT](https://huggingface.co/spaces/prithivMLmods/opencaption-4b-vl-sft)**

OpenCaption-4B-VL-SFT is an advanced vision-language captioning and dense scene-understanding terminal application powered by `prithivMLmods/OpenCaption-4B-VL-SFT-v1.0`. Built upon the Qwen-VL architecture, the model delivers fine-grained, structured visual descriptions covering lighting conditions, spatial compositions, subjects, backgrounds, materials, and thematic summaries.

To maintain safety standards, the application incorporates a dedicated GPU-accelerated pre-screening guard layer using `prithivMLmods/ImageShield-MMCF-0.8B` (`ncii_vision_guard.py`) to classify and block Non-Consensual Intimate Imagery (NCII) and NSFW inputs before caption generation. The interface is served as an interactive retro-styled terminal SPA built with FastAPI and `gradio.Server`.

### **Key Features**

* **Dense Structured Captioning:** Automatically breaks image descriptions into standardized thematic sections—identifying shot types, background elements, foreground details, lighting, and overarching narrative summaries.
* **Integrated Vision Safety Guard:** Utilizes `ImageShield-MMCF-0.8B` to verify image safety prior to model ingestion, preventing the generation of explicit or non-consensual content.
* **Token Streaming Engine:** Employs `TextIteratorStreamer` within threaded execution blocks for responsive real-time generation previews.
* **Retro Terminal Web Interface:** A lightweight, single-page command-line interface supporting keyboard commands (`upload`, `generate`, `copy`, `clear`, `help`), CRT visual effects, Markdown formatting, and one-click example loading.
* **Optimized Inference:** Configured with BF16 precision and automatic memory cleanup routines for seamless operation across modern CUDA and ZeroGPU environments.

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
├── ncii_vision_guard.py
├── pre-requirements.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock
```

### **Installation and Requirements**

To set up the OpenCaption-4B-VL-SFT environment locally, configure your system according to the specifications below. A modern CUDA-enabled GPU is required.

* **Python Version:** Python **3.10** or higher is required; Python **3.12** is recommended.
* **PyTorch Version:** `torch==2.11.0` or above is recommended for optimal compatibility with vision-language transformer kernels.
* **CUDA Version:** **CUDA 12.8+ / 13.0** is recommended (`--extra-index-url [https://download.pytorch.org/whl/cu130](https://download.pytorch.org/whl/cu130)`).

#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust. It ensures rapid virtual environment setup and deterministic dependency management.

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

**Step 4 — Run the application**

```bash
uv run app.py
```

#### **Standard PIP Implementation**

**1. Upgrade Package Manager**

```bash
pip install "pip>=26.1.2"

```

**2. Install Core Dependencies**

```bash
pip install -r requirements.txt
```

#### **Core Requirements List (`requirements.txt`)**

```text
--extra-index-url https://download.pytorch.org/whl/cu130

accelerate==1.14.0
peft==0.20.0
transformers-stream-generator==0.0.5
transformers==5.16.1
qwen-vl-utils==0.0.14
sentencepiece==0.2.2
opencv-python==5.0.0.93
torchvision==0.26.0
matplotlib==3.10.9
einops==0.8.2
spaces==0.51.1
pillow==12.3.0
kernels==0.16.0
gradio==6.25.0
torch==2.11.0
timm==1.0.28
av==17.1.0
```

### **Usage**

Once initialized, access the terminal interface at `http://127.0.0.1:7860/` in your browser.

1. **Upload an Image:** Click the **[ Upload ]** button or type `upload` into the terminal prompt to select a local image.
2. **Run Safety Scan:** The system automatically executes `check_safety` via `ImageShield-MMCF-0.8B`. If flagged as unsafe or NCII, the input is immediately discarded.
3. **Generate Caption:** Type `generate` or click **[ Generate ]** to stream dense visual captions into the terminal.
4. **Copy Output:** Use the `copy` command or click **[ Copy ]** to place the generated Markdown text directly onto your clipboard.

### **License and Source**

* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/opencaption-4b-vl-sft/blob/main/LICENSE.txt)
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/opencaption-4b-vl-sft.git](https://github.com/PRITHIVSAKTHIUR/opencaption-4b-vl-sft.git)
* **Hugging Face Live Space:** [https://huggingface.co/spaces/prithivMLmods/opencaption-4b-vl-sft](https://huggingface.co/spaces/prithivMLmods/opencaption-4b-vl-sft)
