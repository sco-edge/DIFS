import io
import os

import grpc
from concurrent import futures

import torch
from diffusers import StableDiffusionPipeline

from protos.internal import diffusion_service_pb2
from protos.internal import diffusion_service_pb2_grpc


class DiffusionServiceImpl(diffusion_service_pb2_grpc.DiffusionServiceServicer):
    """
    INFaaS diffusion backend using Stable Diffusion.

    Exposes the gRPC interface defined in protos/internal/diffusion_service.proto:
      rpc Generate(DiffusionRequest) returns (DiffusionReply)
    """

    def __init__(self):
        # Select model and device from env vars or defaults.
        model_id = os.environ.get("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")
        device = os.environ.get("SD_DEVICE", "cuda")

        # Optional: half precision for speed/memory.
        dtype = torch.float16 if os.environ.get("SD_USE_FP16", "1") == "1" else torch.float32

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype
        ).to(device)

        # Optional: enable attention optimizations if available.
        if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                # Fallback silently if xformers is not installed
                pass

        self.device = device

    def Generate(self, request, context):
        """
        Handle one diffusion generation request.
        """
        prompt = request.prompt
        if not prompt:
            context.set_details("Prompt must not be empty")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return diffusion_service_pb2.DiffusionReply()

        # Defaults if client sends zeros.
        steps = request.steps if request.steps > 0 else 20
        guidance = request.guidanceScale if request.guidanceScale > 0 else 7.5
        width = request.width if request.width > 0 else 512
        height = request.height if request.height > 0 else 512

        # Seed for reproducibility (0 = no specific seed).
        generator = None
        if request.seed != 0:
            generator = torch.Generator(device=self.device).manual_seed(request.seed)

        # Do generation
        with torch.autocast(self.device if self.device.startswith("cuda") else "cpu"):
            result = self.pipe(
                prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                generator=generator,
            )

        image = result.images[0]

        # Encode to PNG bytes
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        return diffusion_service_pb2.DiffusionReply(
            image_png=png_bytes,
            width=width,
            height=height,
        )


def serve():
    """
    gRPC server entrypoint.
    """
    # Port can be overridden by env var, consistent with INFaaS style.
    port = int(os.environ.get("DIFFUSION_GRPC_PORT", "50052"))

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    diffusion_service_pb2_grpc.add_DiffusionServiceServicer_to_server(
        DiffusionServiceImpl(), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"[DiffusionService] Listening on 0.0.0.0:{port}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
