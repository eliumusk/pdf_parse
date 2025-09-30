from __future__ import annotations
import os
import re
import tempfile
import base64
import io
from pathlib import Path
from typing import Optional, List
from PIL import Image
import requests

from .base import Parser, ParsedDoc
from dotenv import load_dotenv
load_dotenv()


def _extract_title_from_markdown(md: str) -> Optional[str]:
    """Extract title from markdown text"""
    for line in md.splitlines()[:50]:
        if line.startswith("# "):
            t = line[2:].strip()
            if t:
                return t
    # fallback: first non-empty line
    for line in md.splitlines():
        s = line.strip()
        if s and len(s) > 10:  # Avoid short headers/footers
            return s
    return None


def _extract_abstract_from_markdown(md: str) -> Optional[str]:
    """Extract abstract from markdown text"""
    # Look for 'Abstract' section in markdown
    m = re.search(
        r"(?is)^(?:##|#)?\s*abstract\s*\n+(.+?)(?:\n\s*\n|\n(?:##|#)\s+|\Z)",
        md,
        re.MULTILINE
    )
    if m:
        return m.group(1).strip()
    return None


def _pdf_to_images(pdf_path: str, dpi: int = 200) -> List[Image.Image]:
    """
    Convert PDF to list of PIL Images

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for rendering (default 200 for good quality)

    Returns:
        List of PIL Image objects, one per page
    """
    try:
        # Try using PyMuPDF (fitz) first - faster and no external dependencies
        import fitz

        doc = fitz.open(pdf_path)
        images = []

        # Calculate zoom factor from DPI (72 is default PDF DPI)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img_data = pix.tobytes("png")
            from io import BytesIO
            img = Image.open(BytesIO(img_data))
            images.append(img)

        doc.close()
        return images

    except ImportError:
        # Fallback to pdf2image if PyMuPDF not available
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_path, dpi=dpi)
            return images
        except ImportError:
            raise RuntimeError(
                "Neither PyMuPDF (fitz) nor pdf2image is available. "
                "Please install one of them: pip install pymupdf OR pip install pdf2image"
            )


class MinerUVLMParser(Parser):
    """
    MinerU VLM Parser using MinerU2.5 Vision-Language Model

    This parser uses the MinerU2.5-2509-1.2B model to extract content from PDFs.
    It converts PDF pages to images and uses a vision-language model for understanding.

    Supports two modes:
    1. Local mode: Run model locally (requires GPU)
    2. Remote mode: Call H200 HTTP API (recommended for ECS)

    Requirements:
        - PyMuPDF (pymupdf) or pdf2image for PDF to image conversion
        - requests (for remote mode)
        - mineru-vl-utils[transformers] or mineru-vl-utils[vllm] (for local mode)

    Args:
        backend: VLM backend to use
            - "http": Use remote HTTP API (recommended for ECS)
            - "transformers": Use HuggingFace transformers (local)
            - "vllm-engine": Use vLLM engine (local, faster)
            - "vllm-async-engine": Use async vLLM engine (local)
        model_name: Model name or path (for local mode)
        device: Device to run model on (for local mode: "auto", "cuda", "cpu")
        dpi: Resolution for PDF to image conversion (default: 200)
        api_url: HTTP API URL (for remote mode, e.g., "http://172.26.xxx.xxx:30100")
    """

    name = "mineru_vlm"
    version = "2"  # Version 2: Added HTTP API support

    def __init__(
        self,
        *,
        backend: str = "http",  # Default to HTTP mode
        model_name: str = "OpenDataLab/MinerU2.5-2509-1.2B",
        device: str = "auto",
        dpi: int = 200,
        api_url: str = None,  # HTTP API URL for remote mode
        vllm_engine_path: str = None,  # Path to pre-loaded vLLM engine pickle file
        gpu_memory_utilization: float = 0.5,  # GPU memory utilization for vLLM
        tensor_parallel_size: int = 1,  # Number of GPUs for tensor parallelism
    ):
        self.backend = backend
        self.model_name = model_name
        self.device = device
        self.dpi = dpi
        self.api_url = api_url or os.getenv("MINERU_API_URL", "http://127.0.0.1:30100")
        self.vllm_engine_path = vllm_engine_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self._client = None  # Lazy initialization (for local mode)

    def _initialize_client(self):
        """Lazy initialization of MinerU VLM client (for local mode only)"""
        if self._client is not None:
            return

        # HTTP mode doesn't need client initialization
        if self.backend == "http":
            return

        try:
            from mineru_vl_utils import MinerUClient
        except ImportError:
            raise RuntimeError(
                "mineru-vl-utils is required for local VLM parser. "
                "Install with: pip install 'mineru-vl-utils[transformers]' "
                "or pip install 'mineru-vl-utils[vllm]'"
            )

        if self.backend == "transformers":
            # Use transformers backend
            try:
                from modelscope import AutoProcessor, Qwen2VLForConditionalGeneration
            except ImportError:
                raise RuntimeError(
                    "modelscope is required for transformers backend. "
                    "Install with: pip install modelscope"
                )

            print(f"Loading MinerU2.5 model: {self.model_name}")
            print(f"Device: {self.device}")

            # Load model and processor
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                dtype="auto",  # Use torch_dtype for older transformers
                device_map=self.device
            )

            processor = AutoProcessor.from_pretrained(
                self.model_name,
                use_fast=True
            )

            # Initialize client
            self._client = MinerUClient(
                backend="transformers",
                model=model,
                processor=processor
            )

            print("✅ MinerU VLM client initialized successfully")

        elif self.backend in ["vllm-engine", "vllm-async-engine"]:
            # Use vLLM backend
            print(f"Initializing vLLM backend: {self.backend}")
            self._client = MinerUClient(
                backend=self.backend,
                model_name=self.model_name
            )
            print("✅ MinerU VLM client initialized successfully")

        else:
            raise ValueError(
                f"Unsupported backend: {self.backend}. "
                f"Choose from: http, transformers, vllm-engine, vllm-async-engine"
            )

    def _extract_via_http(self, image: Image.Image) -> str:
        """
        Extract content from image via HTTP API

        Args:
            image: PIL Image object

        Returns:
            Extracted content as string
        """
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Call HTTP API
        try:
            response = requests.post(
                f"{self.api_url}/v1/extract",
                json={"image_base64": img_base64},
                timeout=60  # 60 seconds timeout
            )
            response.raise_for_status()

            result = response.json()
            if result.get("success"):
                return result.get("content", "")
            else:
                error_msg = result.get("error", "Unknown error")
                raise RuntimeError(f"API returned error: {error_msg}")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Failed to call MinerU API at {self.api_url}: {e}\n"
                f"Make sure the H200 API service is running and accessible."
            )

    def parse(self, pdf_path: str) -> ParsedDoc:
        """
        Parse PDF using MinerU VLM model

        Process:
        1. Convert PDF to images (one per page)
        2. Use VLM model to extract content from each page
        3. Combine results into markdown
        4. Extract title and abstract
        """
        # Initialize client on first use
        self._initialize_client()

        print(f"\n📄 Parsing PDF: {Path(pdf_path).name}")

        # Step 1: Convert PDF to images
        print(f"🖼️  Converting PDF to images (DPI={self.dpi})...")
        images = _pdf_to_images(pdf_path, dpi=self.dpi)
        page_count = len(images)
        print(f"✅ Converted {page_count} pages to images")

        # Step 2: Extract content from each page using VLM
        if self.backend == "http":
            print(f"🤖 Extracting content using MinerU2.5 VLM (via HTTP API: {self.api_url})...")
        else:
            print(f"🤖 Extracting content using MinerU2.5 VLM (local mode)...")

        all_page_contents = []

        for page_num, image in enumerate(images, start=1):
            print(f"   Processing page {page_num}/{page_count}...", end=" ")

            try:
                # Use HTTP API or local client
                if self.backend == "http":
                    # Remote mode: call HTTP API
                    content = self._extract_via_http(image)
                    page_md = f"\n\n<!-- Page {page_num} -->\n\n{content}"
                else:
                    # Local mode: use two_step_extract
                    extracted_blocks = self._client.two_step_extract(image)
                    # Convert extracted blocks to markdown
                    page_md = self._blocks_to_markdown(extracted_blocks, page_num)

                all_page_contents.append(page_md)
                print("✅")

            except Exception as e:
                print(f"⚠️  Error: {e}")
                # Add placeholder for failed page
                all_page_contents.append(f"\n\n<!-- Page {page_num}: Extraction failed -->\n\n")

        # Step 3: Combine all pages
        fulltext_markdown = "\n\n".join(all_page_contents).strip()

        # Step 4: Extract title and abstract
        title = _extract_title_from_markdown(fulltext_markdown)
        abstract = _extract_abstract_from_markdown(fulltext_markdown)

        print(f"✅ Parsing complete!")
        print(f"   Title: {title[:50] + '...' if title and len(title) > 50 else title}")
        print(f"   Abstract: {'Found' if abstract else 'Not found'}")
        print(f"   Total length: {len(fulltext_markdown)} characters")

        return ParsedDoc(
            title=title,
            abstract=abstract,
            fulltext_markdown=fulltext_markdown,
            page_count=page_count,
        )

    def _blocks_to_markdown(self, blocks: List[dict], page_num: int) -> str:
        """
        Convert extracted blocks to markdown format

        Args:
            blocks: List of extracted content blocks from VLM
            page_num: Page number for reference

        Returns:
            Markdown string for this page
        """
        if not blocks:
            return f"\n\n<!-- Page {page_num}: No content extracted -->\n\n"

        md_lines = [f"\n\n<!-- Page {page_num} -->\n"]

        for block in blocks:
            block_type = block.get("type", "text")
            content = block.get("content", "")

            if block_type == "title":
                md_lines.append(f"# {content}\n")
            elif block_type == "heading":
                md_lines.append(f"## {content}\n")
            elif block_type == "text":
                md_lines.append(f"{content}\n")
            elif block_type == "formula":
                # LaTeX formula
                latex = block.get("latex", content)
                md_lines.append(f"$${latex}$$\n")
            elif block_type == "table":
                # Table in markdown format
                md_lines.append(f"{content}\n")
            elif block_type == "figure":
                # Figure reference
                caption = block.get("caption", "")
                md_lines.append(f"![Figure]({caption})\n")
            else:
                # Unknown type, treat as text
                md_lines.append(f"{content}\n")

        return "\n".join(md_lines)

