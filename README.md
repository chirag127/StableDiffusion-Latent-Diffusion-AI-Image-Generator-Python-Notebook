<div align="center">
  <img src="https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-diffusion-logo.png" alt="Stable Diffusion Logo" width="200"/>
  <h1>StableDiffusion-V1-PyTorch-Reference-Implementation</h1>
  
  <p>A seminal PyTorch implementation of the Stable Diffusion v1 model. This reference showcases the core latent text-to-image diffusion architecture for high-resolution 512x512 image synthesis.</p>

  <!-- Badges -->
  <p>
    <a href="https://github.com/chirag127/StableDiffusion-V1-PyTorch-Reference-Implementation/actions/workflows/ci.yml">
      <img src="https://img.shields.io/github/actions/workflow/status/chirag127/StableDiffusion-V1-PyTorch-Reference-Implementation/ci.yml?branch=main&style=flat-square&label=Build%20Status" alt="Build Status">
    </a>
    <a href="https://codecov.io/gh/chirag127/StableDiffusion-V1-PyTorch-Reference-Implementation">
      <img src="https://img.shields.io/codecov/c/github/chirag127/StableDiffusion-V1-PyTorch-Reference-Implementation?style=flat-square&label=Code%20Coverage" alt="Code Coverage">
    </a>
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python" alt="Python Version">
    <img src="https://img.shields.io/badge/PyTorch-2.x-red?style=flat-square&logo=pytorch" alt="PyTorch Version">
    <img src="https://img.shields.io/badge/HuggingFace-Diffusers-orange?style=flat-square&logo=huggingface" alt="Hugging Face Diffusers">
    <img src="https://img.shields.io/badge/Linting-Ruff-black?style=flat-square&logo=ruff" alt="Linting: Ruff">
    <img src="https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey?style=flat-square" alt="License">
    <a href="https://github.com/chirag127/StableDiffusion-V1-PyTorch-Reference-Implementation">
      <img src="https://img.shields.io/github/stars/chirag127/StableDiffusion-V1-PyTorch-Reference-Implementation?style=flat-square&color=blue" alt="GitHub Stars">
    </a>
  </p>

  <p>
    <a href="https://github.com/chirag127/StableDiffusion-V1-PyTorch-Reference-Implementation/stargazers">
      <img src="https://img.shields.io/badge/Star%20⭐%20this%20Repo-blue?style=social&label=Stars" alt="Star this Repo">
    </a>
  </p>
</div>

---

## 🚀 Overview

This repository provides a high-fidelity, meticulously crafted PyTorch reference implementation of the original Stable Diffusion v1 model. It serves as a foundational resource for understanding the intricate workings of latent text-to-image diffusion, enabling researchers and developers to delve into the core architecture responsible for generating stunning 512x512 resolution images from textual prompts.

The implementation emphasizes clarity, modularity, and adherence to best practices, making it an invaluable tool for both educational purposes and as a robust baseline for further research and development in generative AI.

---

## 🧠 Architecture

The Stable Diffusion v1 model is a latent diffusion model, composed of three primary components working in concert:

1.  **CLIP Text Encoder:** Transforms input text prompts into a rich, semantic latent representation. This encoder extracts the contextual information crucial for guiding the image generation process.
2.  **U-Net Denoising Model:** The heart of the diffusion process, this neural network iteratively removes noise from a latent image representation. It is conditioned on the text embeddings from the CLIP encoder to steer the image generation towards the desired content.
3.  **Variational Autoencoder (VAE):** Responsible for converting raw images into a lower-dimensional latent space and vice-versa. The diffusion process occurs entirely within this compressed latent space, significantly improving computational efficiency.

mermaid
graph TD
    A[Text Prompt] --> B(CLIP Text Encoder)
    B --> C{Text Embedding}
    D[Random Noise Latent] --> E(U-Net Denoising Model)
    C --> E
    E --> F{Denoised Latent}
    F --> G(VAE Decoder)
    G --> H[Generated Image]
    I[Image] --> J(VAE Encoder)
    J --> D


---

## 📋 Table of Contents

*   [🚀 Overview](#-overview)
*   [🧠 Architecture](#-architecture)
*   [📋 Table of Contents](#-table-of-contents)
*   [🤖 AI Agent Directives](#-ai-agent-directives)
*   [⚙️ Setup and Installation](#%EF%B8%8F-setup-and-installation)
*   [📖 Usage](#-usage)
*   [🛠️ Development Standards](#%EF%B8%8F-development-standards)
*   [🤝 Contributing](#-contributing)
*   [📄 License](#-license)

---

## 🤖 AI Agent Directives

<details>
<summary><b>For Elite AI Agents: Operational Mandate (December 2025 Standard)</b></summary>

# SYSTEM: APEX TECHNICAL AUTHORITY & ELITE ARCHITECT (DECEMBER 2025 EDITION)

## 1. IDENTITY & PRIME DIRECTIVE
**Role:** You are a Senior Principal Software Architect and Master Technical Copywriter with **40+ years of elite industry experience**. You operate with absolute precision, enforcing FAANG-level standards and the wisdom of "Managing the Unmanageable."
**Context:** Current Date is **December 2025**. You are building for the 2026 standard.
**Output Standard:** Deliver **EXECUTION-ONLY** results. No plans, no "reporting"—only executed code, updated docs, and applied fixes.
**Philosophy:** "Zero-Defect, High-Velocity, Future-Proof."

---

## 2. INPUT PROCESSING & COGNITION
*   **SPEECH-TO-TEXT INTERPRETATION PROTOCOL:**
    *   **Context:** User inputs may contain phonetic errors (homophones, typos).
    *   **Semantic Correction:** **STRICTLY FORBIDDEN** from executing literal typos. You must **INFER** technical intent based on the project context.
    *   **Logic Anchor:** Treat the `README.md` as the **Single Source of Truth (SSOT)**.
*   **MANDATORY MCP INSTRUMENTATION:**
    *   **No Guessing:** Do not hallucinate APIs.
    *   **Research First:** Use `linkup`/`brave` to search for **December 2025 Industry Standards**, **Security Threats**, and **2026 UI Trends**.
    *   **Validation:** Use `docfork` to verify *every* external API signature.
    *   **Reasoning:** Engage `clear-thought-two` to architect complex flows *before* writing code.

---

## 3. CONTEXT-AWARE APEX TECH STACKS (LATE 2025 STANDARDS)
**Directives:** This repository, `StableDiffusion-V1-PyTorch-Reference-Implementation`, is a seminal PyTorch implementation of the Stable Diffusion v1 model.

*   **PRIMARY SCENARIO: DATA / AI / DEEP LEARNING (Python/PyTorch)**
    *   **Stack:** This project leverages **Python 3.10+**. Key tools include **uv** (for package management and dependency resolution), **Ruff** (for ultra-fast linting and formatting), and **Pytest** (for robust unit and integration testing). Core deep learning components are built with **PyTorch 2.x** and utilize the **Hugging Face Accelerate** and **Diffusers** libraries for efficient model training, inference, and pipeline management.
    *   **Architecture:** Adheres to a **Modular Monolith** pattern, encapsulating distinct components of the latent diffusion model: **CLIP Text Encoder**, **U-Net Denoising Model**, and **Variational Autoencoder (VAE)**. Emphasizes clear interfaces between these modules, ensuring reusability and maintainability.
    *   **AI Integration:** Focused on high-fidelity text-to-image synthesis using a latent diffusion process. Prioritize computational efficiency, memory optimization (e.g., mixed precision training), and reproducibility. Ensure robust data loading pipelines and comprehensive evaluation metrics.
    *   **Performance Optimization:** Utilize PyTorch's `torch.compile` and `float16` for faster training and inference on modern NVIDIA GPUs.

*   **SECONDARY SCENARIO A: CLI / UTILITIES (Python) - *Applicable for scripts and training/inference tools.***
    *   **Stack:** Python with `argparse` or `Click` for robust command-line interface utilities.
    *   **Usage:** Used for launching training jobs, running inference, and performing model evaluations.

---

## 4. ARCHITECTURAL PATTERNS & PRINCIPLES
*   **SOLID Principles:** Applied rigorously, especially Single Responsibility for model components and Dependency Inversion for data handling.
*   **DRY (Don't Repeat Yourself):** Maximized code reuse for common utilities, data preprocessing, and training loops.
*   **YAGNI (You Ain't Gonna Need It):** Focus on the core reference implementation, avoiding unnecessary complexity.
*   **Hexagonal Architecture (Ports & Adapters):** Conceptual separation of core model logic from infrastructure concerns (e.g., data loading, logging, hardware acceleration).
*   **Modular Monolith:** Ensures logical separation of concerns within a single deployable unit.

---

## 5. VERIFICATION & VALIDATION PROTOCOL
*   **UNIT TESTING (Pytest):** Every core function, model layer, and data utility **MUST** have comprehensive unit tests.
    *   **Command:** `uv run pytest`
*   **INTEGRATION TESTING:** Verify interaction between model components (e.g., VAE encode/decode, U-Net forward pass).
*   **DATA VALIDATION:** Ensure input data integrity and correct preprocessing.
*   **MODEL SANITY CHECKS:** Periodically run small inference jobs to ensure model output consistency.
*   **LINTING & FORMATTING (Ruff):**
    *   **Command:** `uv run ruff check .`
    *   **Command:** `uv run ruff format .`
*   **TYPE CHECKING (MyPy):**
    *   **Command:** `uv run mypy .`

---

## 6. SECURITY & RELIABILITY PROTOCOL
*   **DEPENDENCY SCANNING:** Regularly audit dependencies for vulnerabilities (`uv audit`).
*   **CODE REVIEW:** Mandatory for all changes.
*   **ERROR HANDLING:** Robust `try-except` blocks for I/O operations, model loading, and API calls.
*   **RESOURCE MANAGEMENT:** Proper context managers (`with torch.no_grad():`) and explicit memory deallocation where necessary.

--,-

## 7. DEVOPS & CI/CD DIRECTIVES
*   **CI/CD Pipeline (GitHub Actions):** Automate testing, linting, and build verification on every push and pull request.
*   **Containerization (Docker):** Provide Dockerfiles for reproducible environments.
*   **Reproducibility:** Fix random seeds, manage environment variables meticulously.

---

## 8. DOCUMENTATION & COLLABORATION
*   **Docstrings:** Numpydoc or Google-style for all functions, classes, and modules.
*   **Type Hints:** Mandatory for all function signatures.
*   **Contributing Guidelines:** Clear instructions for code contributions and issue reporting.
</details>

---

## ⚙️ Setup and Installation

To get this reference implementation up and running, follow these steps:

1.  **Clone the repository:**
    bash
    git clone https://github.com/chirag127/StableDiffusion-V1-PyTorch-Reference-Implementation.git
    cd StableDiffusion-V1-PyTorch-Reference-Implementation
    

2.  **Install `uv` (if not already installed):**
    `uv` is a highly performant Python package installer and resolver.
    bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    Ensure `uv` is in your PATH.

3.  **Create and activate a virtual environment, then install dependencies:**
    bash
    uv venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    uv sync
    
    *Alternatively, if `pyproject.toml` defines a package:*
    bash
    uv pip install -e .
    

4.  **Download Pre-trained Models:**
    This implementation relies on pre-trained weights for the VAE, U-Net, and CLIP Text Encoder. Instructions for downloading these models (e.g., from Hugging Face Hub) will be provided in a dedicated `models/README.md` or via a script.

---

## 📖 Usage

Once installed, you can explore the Stable Diffusion pipeline through various scripts and Jupyter notebooks.

### Inference from a Text Prompt

To generate an image from a text prompt, use the inference script:

bash
uv run python scripts/inference.py --prompt "a photo of an astronaut riding a horse on mars" --output_path "astronaut_horse.png"


### Training (Fine-tuning)

For fine-tuning the model on a custom dataset, refer to the training scripts:

bash
uv run python scripts/train_text_to_image.py --config configs/default_training.yaml


### Exploring with Jupyter Notebooks

Jupyter notebooks are provided for interactive exploration of individual components and the full pipeline.

bash
uv run jupyter notebook

Open `notebooks/explore_diffusion_pipeline.ipynb` to get started.

---

## 🛠️ Development Standards

This project adheres to the highest development standards to ensure code quality, maintainability, and performance.

### Principles

*   **SOLID Principles:** Applied to ensure modularity and extensibility.
*   **DRY (Don't Repeat Yourself):** Maximizing code reuse and minimizing redundancy.
*   **YAGNI (You Ain't Gonna Need It):** Focusing on essential features for a clear reference implementation.
*   **Readability:** Code is written to be easily understood and debugged.
*   **Performance:** Optimizations for PyTorch (e.g., mixed precision, `torch.compile`) are integrated where appropriate.

### Scripts

The `scripts` directory contains utility functions for common development tasks:

| Script Command                      | Description                                                  |
| :---------------------------------- | :----------------------------------------------------------- |
| `uv run ruff check .`               | Lints the codebase for style and common errors.              |
| `uv run ruff format .`              | Automatically formats the codebase.                          |
| `uv run pytest`                     | Runs all unit and integration tests.                         |
| `uv run mypy .`                     | Performs static type checking across the project.            |
| `uv run python scripts/train.py`    | Initiates model training (with configured parameters).       |
| `uv run python scripts/inference.py`| Runs inference to generate images from text prompts.         |

---

## 🤝 Contributing

Contributions are highly welcome! Please refer to our [CONTRIBUTING.md](.github/CONTRIBUTING.md) for detailed guidelines on how to submit issues, propose features, and contribute code.

---

## 📄 License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0)**. See the [LICENSE](LICENSE) file for details.
