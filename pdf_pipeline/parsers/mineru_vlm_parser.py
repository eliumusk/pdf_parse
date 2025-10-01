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
        api_url: str | None = None,  # Single HTTP API URL
        api_urls: List[str] | None = None,  # Multiple HTTP API URLs for round-robin
        vllm_engine_path: str = None,  # Path to pre-loaded vLLM engine pickle file
        gpu_memory_utilization: float = 0.5,  # GPU memory utilization for vLLM
        tensor_parallel_size: int = 1,  # Number of GPUs for tensor parallelism
        batch_size: int = 32,  # Number of pages to process in one HTTP request
        request_timeout: int = 600,  # HTTP request timeout in seconds
        retry_attempts: int = 3,  # Number of retry attempts for failed requests
        retry_delay: float = 2.0,  # Delay between retries in seconds
    ):
        self.backend = backend
        self.model_name = model_name
        self.device = device
        self.dpi = dpi
        # Support multiple endpoints; fallback to single api_url or env
        if api_urls and len(api_urls) > 0:
            self.api_urls = api_urls
        else:
            self.api_urls = [api_url or os.getenv("MINERU_API_URL", "http://127.0.0.1:30100")]

        # Round-robin index with worker-based offset for better load balancing
        # Each worker process will start from a different endpoint
        self._rr_idx = 0
        self._worker_offset = os.getpid() % len(self.api_urls)

        self.vllm_engine_path = vllm_engine_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.batch_size = batch_size
        self.request_timeout = request_timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
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

    def _extract_via_http(self, image: Image.Image) -> List[dict]:
        """
        Extract structured blocks from image via HTTP API (with retry logic)

        Args:
            image: PIL Image object

        Returns:
            List[dict]: structured blocks as returned by server; fallback to a single text block
        """
        import time

        # Convert image to base64 (PNG to preserve quality)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Retry logic
        last_exception = None
        for attempt in range(self.retry_attempts):
            try:
                # Round-robin select API endpoint with worker-based offset
                # This ensures different workers start from different endpoints
                idx = (self._rr_idx + self._worker_offset) % len(self.api_urls)
                api_url = self.api_urls[idx]
                self._rr_idx += 1

                response = requests.post(
                    f"{api_url}/v1/extract",
                    json={"image_base64": img_base64},
                    timeout=self.request_timeout
                )
                response.raise_for_status()

                result = response.json()
                if result.get("success"):
                    blocks = result.get("blocks")
                    if isinstance(blocks, list):
                        return blocks
                    # Fallback: wrap plain content as a text block
                    content = result.get("content", "")
                    return [{"type": "text", "content": content}]
                else:
                    error_msg = result.get("error", "Unknown error")
                    raise RuntimeError(f"API returned error: {error_msg}")

            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < self.retry_attempts - 1:
                    print(f"⚠️  Request failed (attempt {attempt + 1}/{self.retry_attempts}), retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                    continue

        # All retries failed
        raise RuntimeError(
            f"Failed to call MinerU API after {self.retry_attempts} attempts: {last_exception}\n"
            f"Make sure the H200 API service is running and accessible."
        )

    def _extract_batch_via_http(self, images: List[Image.Image]) -> List[List[dict]]:
        """
        Batch extract structured blocks for multiple images via HTTP API.
        Automatically splits large batches into smaller chunks.
        Returns a list of blocks per page in order.
        """
        import time

        # If batch is too large, split into smaller chunks
        if len(images) > self.batch_size:
            print(f"📦 Splitting {len(images)} pages into batches of {self.batch_size}...")
            all_results = []
            for i in range(0, len(images), self.batch_size):
                batch = images[i:i + self.batch_size]
                batch_results = self._extract_batch_via_http(batch)
                all_results.extend(batch_results)
            return all_results

        # Encode images to base64 PNGs
        payload_images: List[str] = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            payload_images.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

        # Retry logic
        last_exception = None
        for attempt in range(self.retry_attempts):
            try:
                # Round-robin select API endpoint with worker-based offset
                idx = (self._rr_idx + self._worker_offset) % len(self.api_urls)
                api_url = self.api_urls[idx]
                self._rr_idx += 1

                # FIX: Server expects array directly, not {"images": [...]}
                resp = requests.post(
                    f"{api_url}/v1/extract_batch",
                    json=payload_images,  # Send array directly
                    timeout=self.request_timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                blocks_per_page: List[List[dict]] = []
                for item in results:
                    if item.get("success"):
                        blocks = item.get("blocks")
                        if isinstance(blocks, list):
                            blocks_per_page.append(blocks)
                        else:
                            content = item.get("content", "")
                            blocks_per_page.append([{ "type": "text", "content": content }])
                    else:
                        blocks_per_page.append([{ "type": "text", "content": "" }])
                return blocks_per_page

            except requests.exceptions.Timeout as e:
                last_exception = e
                if attempt < self.retry_attempts - 1:
                    print(f"⚠️  Batch request TIMEOUT (attempt {attempt + 1}/{self.retry_attempts})")
                    print(f"   Batch size: {len(payload_images)} pages, Timeout: {self.request_timeout}s")
                    print(f"   Tip: Increase request_timeout or reduce batch_size")
                    print(f"   Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                    continue
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < self.retry_attempts - 1:
                    print(f"⚠️  Batch request failed (attempt {attempt + 1}/{self.retry_attempts}): {type(e).__name__}")
                    print(f"   Error: {str(e)[:200]}")
                    print(f"   Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                    continue

        # All retries failed
        error_type = type(last_exception).__name__
        raise RuntimeError(
            f"Failed to call MinerU batch API after {self.retry_attempts} attempts.\n"
            f"Error type: {error_type}\n"
            f"Last error: {last_exception}\n"
            f"Batch size: {len(payload_images)} pages, Timeout: {self.request_timeout}s\n"
            f"Tip: If timeout errors, increase request_timeout in config (recommended: 600s for batch_size=32)"
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
        import time
        extract_start = time.time()

        if self.backend == "http":
            print(f"🤖 Extracting content using MinerU2.5 VLM (via HTTP API, batch_size={self.batch_size})...")
            print(f"   📡 Load balancing across {len(self.api_urls)} endpoints")
            print(f"   🔀 Worker PID={os.getpid()}, starting from endpoint index {self._worker_offset}")
        else:
            print("🤖 Extracting content using MinerU2.5 VLM (local mode)...")

        all_page_contents = []

        if self.backend == "http":
            # Prefer batch mode to reduce HTTP overhead
            try:
                blocks_pages = self._extract_batch_via_http(images)
                for page_num, blocks in enumerate(blocks_pages, start=1):
                    page_md = self._blocks_to_markdown(blocks, page_num)
                    all_page_contents.append(page_md)

                extract_time = time.time() - extract_start
                pages_per_sec = page_count / extract_time if extract_time > 0 else 0
                print(f"   ⚡ Extraction speed: {pages_per_sec:.2f} pages/sec ({extract_time:.1f}s total)")

            except Exception as e:
                print(f"⚠️  Batch processing failed: {e}")
                print("   Falling back to per-page requests...")
                # Fallback to per-page requests
                for page_num, image in enumerate(images, start=1):
                    try:
                        blocks = self._extract_via_http(image)
                        page_md = self._blocks_to_markdown(blocks, page_num)
                        all_page_contents.append(page_md)
                    except Exception as ee:
                        print(f"   ❌ Page {page_num} failed: {ee}")
                        all_page_contents.append(f"\n\n<!-- Page {page_num}: Extraction failed: {ee} -->\n\n")
        else:
            for page_num, image in enumerate(images, start=1):
                try:
                    blocks = self._client.two_step_extract(image)
                    page_md = self._blocks_to_markdown(blocks, page_num)
                    all_page_contents.append(page_md)
                except Exception as e:
                    all_page_contents.append(f"\n\n<!-- Page {page_num}: Extraction failed: {e} -->\n\n")

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

