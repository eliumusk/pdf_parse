#!/usr/bin/env python3
"""
Deploy MinerU VLM model on H200 using vLLM with HTTP API
使用 vLLM 在 H200 上部署 MinerU 模型，并提供 HTTP API 服务

使用方法：
在 H200 上运行:
    python deploy_mineru_vllm_h200.py --gpu-id 7 --gpu-memory 0.3 --port 30100

架构说明：
    ECS (PDF处理) → HTTP请求 → H200 (模型推理) → 返回结果 → ECS
"""

import os
import argparse
import io
import base64
import asyncio
from typing import List, Dict, Any
from PIL import Image

from vllm import LLM
from mineru_vl_utils import MinerUClient

# FastAPI for HTTP service
try:
    from fastapi import FastAPI, HTTPException, File, UploadFile
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("⚠️  FastAPI not installed. Installing...")
    os.system("pip install fastapi uvicorn python-multipart -q")
    from fastapi import FastAPI, HTTPException, File, UploadFile
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn

# Try to import MinerULogitsProcessor (for vllm>=0.10.1)
try:
    from mineru_vl_utils import MinerULogitsProcessor
    HAS_LOGITS_PROCESSOR = True
except ImportError:
    HAS_LOGITS_PROCESSOR = False
    print("⚠️  MinerULogitsProcessor not available (vllm<0.10.1)")


# ============================================================================
# Global variables for model
# ============================================================================
global_llm = None
global_client = None


# ============================================================================
# FastAPI Request/Response Models
# ============================================================================
class ExtractRequest(BaseModel):
    """Request model for image extraction"""
    image_base64: str  # Base64 encoded image

class ExtractResponse(BaseModel):
    """Response model for extraction"""
    success: bool
    content: str = ""
    blocks: list | None = None  # structured blocks, preferred
    error: str = ""


# ============================================================================
# FastAPI App
# ============================================================================
app = FastAPI(
    title="MinerU VLM API",
    description="MinerU2.5-2509-1.2B Vision-Language Model API for PDF extraction",
    version="1.0.0"
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok" if global_client is not None else "not_ready",
        "model": "MinerU2.5-2509-1.2B",
        "backend": "vllm-engine"
    }


@app.post("/v1/extract", response_model=ExtractResponse)
async def extract_from_image(request: ExtractRequest):
    """
    Extract content from image

    Request body:
    {
        "image_base64": "base64_encoded_image_data"
    }

    Returns:
    {
        "success": true,
        "content": "extracted markdown content",
        "error": ""
    }
    """
    if global_client is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Extract content using MinerU client
        result = global_client.two_step_extract(image)

        # Prefer returning structured blocks
        if isinstance(result, list):
            return ExtractResponse(
                success=True,
                content="",
                blocks=result,
                error=""
            )
        else:
            # Fallback: return as plain text block if structure missing
            return ExtractResponse(
                success=True,
                content=str(result),
                blocks=None,
                error=""
            )

    except Exception as e:
        return ExtractResponse(
            success=False,
            content="",
            error=str(e)
        )


@app.post("/v1/extract_batch")
async def extract_batch(images: List[str]):
    """
    Extract content from multiple images (batch processing)

    Request body:
    {
        "images": ["base64_image_1", "base64_image_2", ...]
    }
    """
    if global_client is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    for idx, image_base64 in enumerate(images):
        try:
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            result = global_client.two_step_extract(image)

            if isinstance(result, list):
                results.append({
                    "index": idx,
                    "success": True,
                    "content": "",
                    "blocks": result,
                    "error": ""
                })
            else:
                results.append({
                    "index": idx,
                    "success": True,
                    "content": str(result),
                    "blocks": None,
                    "error": ""
                })
        except Exception as e:
            results.append({
                "index": idx,
                "success": False,
                "content": "",
                "blocks": None,
                "error": str(e)
            })

    return {"results": results}


# ============================================================================
# Model Deployment
# ============================================================================
def deploy_mineru_vllm(
    model_path: str,
    gpu_memory_utilization: float = 0.3,
    tensor_parallel_size: int = 1,
    gpu_id: int = 7,
):
    """
    Deploy MinerU VLM model using vLLM engine

    Args:
        model_path: Path to the model
        gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
        tensor_parallel_size: Number of GPUs to use
        gpu_id: Which GPU to use (0-7)
    """
    global global_llm, global_client

    # Set CUDA_VISIBLE_DEVICES to use only the specified GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print("="*70)
    print("🚀 Deploying MinerU VLM on H200")
    print("="*70)
    print(f"   Model: {model_path}")
    print(f"   GPU: {gpu_id}")
    print(f"   GPU Memory Utilization: {gpu_memory_utilization * 100}%")
    print(f"   Tensor Parallel Size: {tensor_parallel_size}")
    print("="*70)
    
    # Initialize vLLM engine
    print("\n📦 Loading vLLM engine...")

    llm_kwargs = {
        "model": model_path,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": True,
        "max_num_seqs": 512,             # 并发序列
        "max_num_batched_tokens": 16384, # 批大小
        "max_model_len": 4096,           # 上下文长度
        "mm_encoder_tp_mode": "data",    # VLM 优化
        "mm_processor_cache_gb": 8,      # 图像缓存
    }

    # Add logits processor if available
    if HAS_LOGITS_PROCESSOR:
        llm_kwargs["logits_processors"] = [MinerULogitsProcessor]
        print("   ✅ Using MinerULogitsProcessor")

    try:
        global_llm = LLM(**llm_kwargs)
        print("✅ vLLM engine loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load vLLM engine: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Try reducing --gpu-memory (e.g., 0.2 or 0.1)")
        print("   2. Make sure no other processes are using the GPU")
        print("   3. Check GPU memory: nvidia-smi")
        raise

    # Initialize MinerU client
    print("\n🔧 Initializing MinerU client...")
    try:
        global_client = MinerUClient(
            backend="vllm-engine",
            vllm_llm=global_llm
        )
        print("✅ MinerU client initialized!")
    except Exception as e:
        print(f"❌ Failed to initialize MinerU client: {e}")
        raise

    # Test the model
    print("\n🧪 Testing model with a simple image...")

    # Create a simple test image
    test_image = Image.new('RGB', (100, 100), color='white')

    try:
        result = global_client.two_step_extract(test_image)
        print("✅ Model test successful!")
        if isinstance(result, str):
            print(f"   Result preview: {result[:100]}...")
        else:
            print(f"   Result type: {type(result)}")
    except Exception as e:
        print(f"⚠️  Model test failed: {e}")
        print("   (This might be OK if the test image is too simple)")

    print("\n" + "="*70)
    print("🎉 Model deployment complete!")
    print("="*70)

    return global_llm, global_client


def main():
    parser = argparse.ArgumentParser(
        description="Deploy MinerU VLM on H200 with HTTP API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy on GPU 7 with HTTP API on port 30100
  python deploy_mineru_vllm_h200.py --gpu-id 7 --gpu-memory 0.3 --port 30100

  # Use GPU 5 with 20% memory (if GPU 7 is busy)
  python deploy_mineru_vllm_h200.py --gpu-id 5 --gpu-memory 0.2 --port 30100

  # Use 2 GPUs with tensor parallelism
  python deploy_mineru_vllm_h200.py --gpu-id 6 --tensor-parallel 2 --gpu-memory 0.4 --port 30100
        """
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="/mnt/chensiheng/muskliu/model_cache/OpenDataLab/MinerU2___5-2509-1___2B",
        help="Path to the model (default: %(default)s)"
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.3,
        help="Fraction of GPU memory to use, 0.0-1.0 (default: 0.3 = 30%%)"
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=7,
        help="Which GPU to use, 0-7 (default: 7, the empty one)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30100,
        help="HTTP API port (default: 30100)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="HTTP API host (default: 0.0.0.0)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not (0.0 < args.gpu_memory <= 1.0):
        parser.error("--gpu-memory must be between 0.0 and 1.0")

    if not (0 <= args.gpu_id <= 7):
        parser.error("--gpu-id must be between 0 and 7")

    # Deploy model
    print("\n" + "="*70)
    print("🚀 Starting MinerU VLM HTTP API Service")
    print("="*70)

    try:
        deploy_mineru_vllm(
            model_path=args.model_path,
            gpu_memory_utilization=args.gpu_memory,
            tensor_parallel_size=args.tensor_parallel,
            gpu_id=args.gpu_id,
        )
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        return 1

    # Start HTTP API server
    print("\n" + "="*70)
    print("🌐 Starting HTTP API Server")
    print("="*70)
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"\n📡 API Endpoints:")
    print(f"   Health check: http://{args.host}:{args.port}/health")
    print(f"   Extract: http://{args.host}:{args.port}/v1/extract")
    print(f"   Batch extract: http://{args.host}:{args.port}/v1/extract_batch")
    print(f"\n💡 API Documentation: http://{args.host}:{args.port}/docs")
    print("="*70)
    print("\n✅ Service is ready! Press Ctrl+C to stop.\n")

    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")

    return 0


if __name__ == "__main__":
    exit(main())

