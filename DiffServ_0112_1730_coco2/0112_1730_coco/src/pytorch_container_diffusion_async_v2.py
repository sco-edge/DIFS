import asyncio
import grpc
import grpc.aio
from concurrent import futures
import torch
import io
from pathlib import Path

# import the compiled files from the root
import query_pb2
import query_pb2_grpc
import infaas_request_status_pb2

### Merge
from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path
import os
from typing import List, Optional, Tuple
import time

from diffusers import (
    EulerAncestralDiscreteScheduler, 
    DPMSolverMultistepScheduler, 
    LMSDiscreteScheduler, 
    DDIMScheduler
)

import datetime
import secrets

import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
import sys
from scipy import linalg

import argparse  # added
import json      # added

# ===============================================
# 🌟 global constant settings (reflecting the INFaaS folder structure) 🌟
# ===============================================
MODEL_DIRECTORY = '/tmp/model'
OUTPUT_ROOT_DIR = '/tmp/diffusion_output' 
MODEL_FILE_NAME = "sd-v1-4.safetensors"
#MODEL_FILE_NAME = "v1-5-pruned-emaonly.safetensors"
CKPT_FILE_PATH = Path(MODEL_DIRECTORY) / MODEL_FILE_NAME
OUTPUT_DIR = Path(OUTPUT_ROOT_DIR)

# define the model pipeline type (for explicit type hints)
DiffusionPipeline = StableDiffusionPipeline

# [ADDED] JSON load and mapping utilities
def load_json(file_path):
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_filename_by_id(data, target_id):
    for item in data:
        if item.get("id") == target_id:
            return item.get("filename")
    return None

class ModelExecutor(query_pb2_grpc.QueryServicer):

    def __init__(
        self, 
        model_id: int, 
        model_filename: str, 
        sampler_id: int, 
        sampler_name: str, 
        thread_count: int, 
        port: int,           # added port number
        npz_path: str
    ):
        """
        Initialize ModelExecutor and load the model
        """        
        print(f"[INFO] ModelExecutor initialized (Model ID: {model_id}, File: {model_filename})")
        self.model_id = model_id # store the model ID
        self.model_directory = '/tmp/model'
        self.model_file_name = model_filename # use the received filename
        self.ckpt_file_path = Path(self.model_directory) / self.model_file_name
        # store the parameters
        self.sampler_id = sampler_id
        self.sampler_name = sampler_name
        self.thread_count = thread_count
        self.port = port     # store as a class member variable
        self.real_stats_path = npz_path

        # load the model
        self.pipe, self.device = self.load_model()
        self.fid = FrechetInceptionDistance(feature=2048).to(self.device)
        
        if self.pipe is None:
            print("[ERROR] model load failed")
        
        real_stats_path = "/workspace/real_stats.npz"
        if os.path.exists(real_stats_path):
            # compute the real-image mu and sigma values during load. 
            real_stats = np.load(real_stats_path)
            self.mu_real = real_stats['mu']         # (2048, )
            self.sigma_real = real_stats['sigma']   # (2048, 2048)
            print(f"📊 real-image mu value ({self.mu_real})")
            print(f"📊 real-image sigma value ({self.sigma_real})")
        else:
            print("[ERROR] Not Found: /workspace/real_stats.npz")

    def load_model(self) -> Tuple[Optional[StableDiffusionPipeline], str]:
        """
        load a Stable Diffusion model from a single file (.safetensors)
        """

        total_start = time.perf_counter()
        # 1. measure the file-system load
        print(f"[TIMER] stage 1: start loading the weight file... ({self.ckpt_file_path})")
        load_start = time.perf_counter()
        
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
            print(f"cuda")
        else:
            device = "cpu"
            dtype = torch.float32
            print(f"cpu")

        try:
            print(f"[INFO] attempting to load the model: {self.ckpt_file_path}")
            # load the checkpoint using the from_single_file method
            pipe = StableDiffusionPipeline.from_single_file(
                str(self.ckpt_file_path),
                torch_dtype=dtype,
                load_safety_checker=False,
                low_cpu_mem_usage=False, # <--- add this option to prevent lazy loading
            )
            pipe.to(device)
            load_end = time.perf_counter()
            print(f" ✅ [DONE] file load time: {load_end - load_start:.2f}s")
            

            # 2. measure the warm-up time
            # if the model.safetensors progress bar appears again here, this section is the bottleneck.
            print(f"[TIMER] stage 2: start component optimization (warm-up)...")
            warmup_start = time.perf_counter()

            with torch.inference_mode():
                # ignore the safety checker entirely (to avoid NSFW errors)
                pipe.safety_checker = None
                pipe(
                    prompt="a silent dog",
                    num_inference_steps=1,
                    width=128,
                    height=128,
                    guidance_scale=1.0
                )
            
            warmup_end = time.perf_counter()
            print(f" ✅ [DONE] warm-up execution time: {warmup_end - warmup_start:.2f}s")

            total_end = time.perf_counter()
            print("=" * 60)
            print(f"🚀 [FINAL] total time until the server is ready: {total_end - total_start:.2f}s")
            print("=" * 60)

            print(f"[INFO] weight loading 100% complete! GPU is ready")
            return pipe, device
        except Exception as e:
            print(f"[ERROR] an exception occurred while loading the model: {e}")
            return None, device
    
    def QueryOnline(self, request, context):
        """
        Handle gRPC QueryOnline requests
        """
        if self.pipe is None:
            status_msg = infaas_request_status_pb2.InfaasRequestStatus(
                status=infaas_request_status_pb2.UNAVAILABLE,
                msg="Model not loaded"
            )
            return query_pb2.QueryOnlineResponse(status=status_msg)

        print(f"[INFO] request received - prompt: {request.Prompt}")

        # run inference
        try:
            # --- start timing ---
        # record the CPU time after synchronizing, since GPU work may still be pending
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            saved_paths = self.ExecuteInference(
                prompt=list(request.Prompt),
                steps=request.Steps if request.Steps > 0 else 30,
                cfg=request.CFG_Scale if request.CFG_Scale > 0 else 7.5,
                batch_size=request.BatchSize if request.BatchSize > 0 else 1,
                seed=request.Seed,
                #sampler_type=getattr(request, 'SamplerType', 1) # assumes the proto has a SamplerType field
                sampler_type=1 #hardcoded
                #height_px=256, 
                #width_px=256
            )

            # wait until the GPU work is fully finished, then record the end time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            # --- stop timing ---

            elapsed_time = end_time - start_time
            print(f"------------------------------------------------------------")
            print(f"[PERFORMANCE LOG] total inference time: {elapsed_time:.4f} s (CPU+GPU)")
            print(f"------------------------------------------------------------")

            # 2. print the result log
            print(f"[INFO] inference and file saving complete!")
            str_image_paths = []
            for path in saved_paths:
                print(f" >> saved location: {path}")
                # convert the Path object to a string and add it to the list
                str_image_paths.append(str(path))
            
            # 3. build the success status message
            status_msg = infaas_request_status_pb2.InfaasRequestStatus(
                status=infaas_request_status_pb2.SUCCESS
            )

            # 4. return the results in the image_paths field per the modified proto definition
            return query_pb2.QueryOnlineResponse(
                image_path=str_image_paths,
                status=status_msg
            )

        except Exception as e:
            print(f"[ERROR] an error occurred during inference: {e}")
            status_msg = infaas_request_status_pb2.InfaasRequestStatus(
                status=infaas_request_status_pb2.INVALID,
                msg=str(e)
            )
            return query_pb2.QueryOnlineResponse(status=status_msg)

    async def QueryOnlineImage(self, request, context):

        height_px = 512
        width_px  = 512

        if self.pipe is None:
            status_msg = infaas_request_status_pb2.InfaasRequestStatus(
                status=infaas_request_status_pb2.UNAVAILABLE,
                msg="Model not loaded"
            )
            # streaming response via yield (single-shot error response)
            yield query_pb2.QueryOnlineImageResponse(status=status_msg)
            return

        print(f"request received (prompt: {request.Prompt} Steps: {request.Steps} Sampler_Type: {request.Sampler_Type}")

        sampler_type = request.Sampler_Type
        print(f"\n--- image inference started (Sampler Type: {sampler_type}) and saving process started ---")
        # 1. configure the sampler (scheduler)
        if sampler_type == 1:
            # Euler Ancestral (Euler a)
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
            sampler_name = "Euler_a"
        elif sampler_type == 2:
            # DPM++ 2M Karras (very fast and high quality)
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config, use_karras_sigmas=True
            )
            sampler_name = "DPM++_Karras"
        elif sampler_type == 3:
            # LMS (Linear Multi-Step)
            self.pipe.scheduler = LMSDiscreteScheduler.from_config(self.pipe.scheduler.config)
            sampler_name = "LMS"
        elif sampler_type == 4:
            # DDIM
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
            sampler_name = "DDIM"
        else:
            print(f"[WARNING] unknown sampler type ({sampler_type}); using the default (Euler a).")
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
            sampler_name = "Euler_a_Default"
        
        # 2. configure the random number generator (for reproducibility)
        generator = torch.Generator(self.device).manual_seed(request.Seed)

        try:
            # --- start timing ---
            # record the CPU time after synchronizing, since GPU work may still be pending
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            prompts = list(request.Prompt)
            total_count = len(prompts)
            chunk_size = 10  # split into chunks of 10 to protect server memory

             # generate the directory name
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            random_str = secrets.token_hex(3) # 6-character random string
            unique_name = f"{now}_{random_str}"           
            target_dir = OUTPUT_DIR / unique_name
            print(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            
            self.fid.reset() # reset the accumulated fake features for each new request
            all_generated_tensors = []  # for FID computation
            print(f"[SERVER] received {total_count} requests total. starting chunked inference in groups of {chunk_size}...")

            for i in range(0, total_count, chunk_size):
                batch_prompts = prompts[i : i + chunk_size]
                
                # 1. run inference (mini-batch)
                with torch.inference_mode():
                    # maximize memory efficiency using autocast and slicing
                    ###with torch.cuda.amp.autocast():
                    output = self.pipe(
                        prompt=batch_prompts,
                        num_inference_steps=request.Steps,
                        guidance_scale=7.5,
                        #generator=generator
                        batch_size=request.BatchSize if request.BatchSize > 0 else 1,
                        seed=request.Seed,
                        sampler_type=request.Sampler_Type, #hardcoded
                        #height_px=256, 
                        #width_px=256
                        output_path=target_dir
                    )
                
                # 2. process the resulting images and stream them individually
                for j, pil_img in enumerate(output.images):
                    # (1) convert to tensors for FID computation and keep them in a list
                    # convert to a uint8 tensor of shape [C, H, W]
                    img_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    #all_generated_tensors.append(img_tensor)

                    # update the FID object with the fake image (only the feature vectors are stored)
                    self.fid.update(img_tensor, real=False)                    
                    # [MEMORY KEY 2] immediately free tensors that are no longer needed
                    del img_tensor

                    # (2) serialize the image to binary and send it to the client
                    img_io = io.BytesIO()
                    pil_img.save(img_io, format="PNG")
                    img_bytes = img_io.getvalue()

                    # create the output folder and save the image (applying the generate_and_save_image logic)
                    target_dir.mkdir(parents=True, exist_ok=True)
                    print(f"saving the generated images to the '{target_dir}' folder.")

                    # filename format: sd_output_[sampler_name]_[index]_[seed]_[resolution].png
                    filename = target_dir / f"sd_{sampler_name}_b{i+j+1}_s{request.Seed}_{width_px}x{height_px}.png"
                    pil_img.save(filename)
                    print(f"  -> saved: {filename}")
                    #saved_files.append(filename)

                    yield query_pb2.QueryOnlineImageResponse(
                        image_path=f"gen_{i+j}.png",
                        image_data=img_bytes,
                        status=infaas_request_status_pb2.InfaasRequestStatus(status=1)
                    )

            # 3. [KEY] compute the full FID over all 100 images after all loops finish
            #if len(all_generated_tensors) >= total_count:
            print(f"[SERVER] {total_count} images generated. computing FID...")

            # 4. extract mu/sigma from the accumulated fake data
            # compute mu_fake and sigma_fake using fake_features_sum and cov_sum
            num_samples = self.fid.fake_features_num_samples.item()
            mu_fake = (self.fid.fake_features_sum / num_samples).cpu().numpy()
            
            # covariance computation: (E[X^2] - E[X]^2)
            sigma_fake = (self.fid.fake_features_cov_sum.cpu().numpy() / num_samples) - np.outer(mu_fake, mu_fake)

            # 5. [IMPORTANT] compare the pre-loaded mu_real and sigma_real with the freshly computed mu_fake and sigma_fake
            final_fid = self.calculate_fid_from_stats(
                self.mu_real, self.sigma_real, 
                mu_fake, sigma_fake
            )
            
            print(f"🌟 [RESULT] full FID score over {total_count} images: {final_fid:.4f}")         

            """
            # batch the generated images into one [100, 3, H, W]
            generated_batch = torch.cat(all_generated_tensors, dim=0).to(self.device)
            
            # compute FID (assuming the self.fid object is defined)
            # real_features must already be loaded into self.fid.real_features etc.
            self.fid.update(generated_batch, real=False)
            final_fid = self.fid.compute().item()

            fid_score = self.calculate_fid_from_stats(self.mu_real, self.sigma_real, mu_gen, sigma_gen)
            print(f"🔥 final FID Score: {fid_score:.4f}")
            
            print(f"🌟 [RESULT] full FID over 100 images: {final_fid:.4f}")
            """
        except Exception as e:
            print(f"[ERROR] occurred during inference: {str(e)}")
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)

    async def QueryOnlineImage_Old(self, request, context):
        """
        Handle gRPC QueryOnline requests
        """
        if self.pipe is None:
            status_msg = infaas_request_status_pb2.InfaasRequestStatus(
                status=infaas_request_status_pb2.UNAVAILABLE,
                msg="Model not loaded"
            )
            ##return query_pb2.QueryOnlineImageResponse(status=status_msg)
            # streaming response via yield (single-shot error response)
            yield query_pb2.QueryOnlineImageResponse(status=status_msg)
            return

        print(f"[INFO] request received - prompt: {request.Prompt}")
        print(f"[INFO] request.Steps: {request.Steps}")
        print(f"[INFO] request.Sampler_Type: {request.Sampler_Type}")

        # run inference
        try:
            # --- start timing ---
            # record the CPU time after synchronizing, since GPU work may still be pending
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            # generate the directory name
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            random_str = secrets.token_hex(3) # 6-character random string
            unique_name = f"{now}_{random_str}"           
            target_dir = OUTPUT_DIR / unique_name
            print(target_dir)

            saved_paths = self.ExecuteInference(
                prompt=list(request.Prompt),
                steps=request.Steps if request.Steps > 0 else 30,
                cfg=request.CFG_Scale if request.CFG_Scale > 0 else 7.5,
                batch_size=request.BatchSize if request.BatchSize > 0 else 1,
                seed=request.Seed,
                #sampler_type=getattr(request, 'SamplerType', 1) # assumes the proto has a SamplerType field
                sampler_type=request.Sampler_Type, #hardcoded
                #height_px=256, 
                #width_px=256
                output_path=target_dir
            )

            # wait until the GPU work is fully finished, then record the end time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            # --- stop timing ---

            # --- save the npz file ---
            mu_gen, sigma_gen = self.get_statistics(target_dir, self.device)
            # 2. set the filename and path (named gen_stats.npz inside the target_dir folder) 
            out_file = os.path.join(target_dir, "gen_stats.npz")
            np.savez(out_file, mu=mu_gen, sigma=sigma_gen)
            print(f"✅ statistics saved: {out_file}")

            # --- [ADDED] FID computation logic ---
            # path to the real-image statistics file
            # 1. specify the absolute path inside the container directly (most reliable)
            real_stats_path = "/workspace/real_stats.npz"
            fid_score = None

            if os.path.exists(real_stats_path):
                #print(f"📊 starting FID computation (reference: {real_stats_path})")
                # 1. load the real-image statistics
                #real_stats = np.load(real_stats_path)
                #mu_real = real_stats['mu']
                #sigma_real = real_stats['sigma']

                # 2. compute FID by calling the provided function
                fid_score = self.calculate_fid_from_stats(self.mu_real, self.sigma_real, mu_gen, sigma_gen)
                print(f"🔥 final FID Score: {fid_score:.4f}")
            else:
                print(f"⚠️ WARNING: could not find {real_stats_path}; skipping FID computation.")

            elapsed_time = end_time - start_time
            print(f"------------------------------------------------------------")
            print(f"[PERFORMANCE LOG] total inference time: {elapsed_time:.4f} s (CPU+GPU)")
            if fid_score is not None:
                print(f"[PERFORMANCE LOG] FID score: {fid_score:.4f}")
            print(f"------------------------------------------------------------")

            # 2. print the result log
            print(f"[INFO] inference and file saving complete!")
            str_image_paths = []
            raw_images_bytes = []
            for path in saved_paths:
                print(f" >> saved location: {path}")
                # convert the Path object to a string and add it to the list
                str_image_paths.append(str(path))
                
                # open the file in binary read mode ('rb') and extract the byte data
                with open(path, "rb") as f:
                    #raw_data.append(f.read())
                    image_binary = f.read()

                # success status message (include in every response or only the last, depending on the proto design)
                status_msg = infaas_request_status_pb2.InfaasRequestStatus(
                    status=infaas_request_status_pb2.SUCCESS
                )
                # [CHANGED] use yield instead of return to send the images one at a time.
                yield query_pb2.QueryOnlineImageResponse(
                    image_path=str(path),   # path of the image currently being sent
                    image_data=image_binary,     # binary data of the image currently being sent
                    status=status_msg
                )
                
                # [SERVER STABILITY] yield during bulk transfer so the loop does not monopolize the event loop
                await asyncio.sleep(0.01)

            print(f"[INFO] finished streaming {len(saved_paths)} images")
            
            # 3. build the success status message
            #status_msg = infaas_request_status_pb2.InfaasRequestStatus(
            #    status=infaas_request_status_pb2.SUCCESS
            #)

            # 4. return the results in the image_paths field per the modified proto definition
            #return query_pb2.QueryOnlineImageResponse(
            #    image_paths=str_image_paths,
            #    image_data=raw_images_bytes,  # modified field
            #    status=status_msg
            #)

        except Exception as e:
            print(f"[ERROR] an error occurred during inference: {e}")
            status_msg = infaas_request_status_pb2.InfaasRequestStatus(
                status=infaas_request_status_pb2.INVALID,
                msg=str(e)
            )
            yield query_pb2.QueryOnlineImageResponse(status=status_msg)
        """
        except Exception as e:
            print(f"[ERROR] an error occurred during inference: {e}")
            status_msg = infaas_request_status_pb2.InfaasRequestStatus(
                status=infaas_request_status_pb2.INVALID,
                msg=str(e)
            )
            return query_pb2.QueryOnlineImageResponse(status=status_msg)
        """
    def ExecuteInference(self, prompt: list, steps: int, cfg: float, batch_size: int, seed: int,
                         sampler_type: int = 1,  # 1: Euler a, 2: DPM++, 3: LMS, 4: DDIM 
                         height_px: int = 384, width_px: int = 384, output_path: Path = OUTPUT_DIR):
        """
        run the actual diffusion inference engine and save the images
        """
        print(f"\n--- image inference started (Sampler Type: {sampler_type}) and saving process started ---")
        
        # 1. configure the sampler (scheduler)
        if sampler_type == 1:
            # Euler Ancestral (Euler a)
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
            sampler_name = "Euler_a"
        elif sampler_type == 2:
            # DPM++ 2M Karras (very fast and high quality)
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config, use_karras_sigmas=True
            )
            sampler_name = "DPM++_Karras"
        elif sampler_type == 3:
            # LMS (Linear Multi-Step)
            self.pipe.scheduler = LMSDiscreteScheduler.from_config(self.pipe.scheduler.config)
            sampler_name = "LMS"
        elif sampler_type == 4:
            # DDIM
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
            sampler_name = "DDIM"
        else:
            print(f"[WARNING] unknown sampler type ({sampler_type}); using the default (Euler a).")
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
            sampler_name = "Euler_a_Default"
        
        # 2. configure the random number generator (for reproducibility)
        generator = torch.Generator(self.device).manual_seed(seed)
        #prompt_list = [prompt] * batch_size

        # 3. validate the prompt list
        # handle the case where fewer prompts are provided than batch_size
        if len(prompt) < batch_size:
            print(f"[WARNING] the number of prompts provided ({len(prompt)}) is less than the BatchSize ({batch_size}); the remainder is filled with the last prompt.")
            prompt = prompt + [prompt[-1]] * (batch_size - len(prompt))
        elif len(prompt) > batch_size:
            print(f"[INFO] using only the first part of the provided prompts to match the BatchSize ({batch_size}).")
            prompt = prompt[:batch_size]

        # 4. generate the images
        print(f"generating {batch_size} images... (Sampler: {sampler_name})")
        print(f"{batch_size} images total ({width_px}x{height_px})")
        output = self.pipe(
            prompt=prompt, # pass the prompt as a list
            num_inference_steps=steps,
            guidance_scale=cfg,
            height=height_px,
            width=width_px,
            generator=generator
        )
        
        generated_images = output.images

        # 5. create the output folder and save the images (applying the generate_and_save_image logic)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"saving the generated images to the '{output_path}' folder.")

        # 5. image saving logic
        output_path.mkdir(parents=True, exist_ok=True)
        saved_files = []
        for i, image in enumerate(generated_images):
            # filename format: sd_output_[sampler_name]_[index]_[seed]_[resolution].png
            filename = output_path / f"sd_{sampler_name}_b{i+1}_s{seed}_{width_px}x{height_px}.png"
            image.save(filename)
            print(f"  -> saved: {filename}")
            saved_files.append(filename)
            print(f" -> saved: {filename} (Prompt: {prompt[i][:20]}...) ")

        print("-" * 50)
        return saved_files # return the list of saved file paths

    # --- 1. function to extract the statistics (mu, sigma) ---
    def get_statistics(self, image_dir, device):
        fid = FrechetInceptionDistance(feature=2048).to(device)
        
        valid_extensions = ('.png', '.jpg', '.jpeg')
        images = []
        files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)])
        
        if not files:
            raise ValueError(f"❌ no images in the folder: {image_dir}")

        print(f"📦 loading {len(files)} images...")
        for filename in files:
            img = Image.open(os.path.join(image_dir, filename)).convert('RGB')
            img = img.resize((299, 299), Image.LANCZOS)
            images.append(np.array(img))

        # (N, H, W, C) -> (N, C, H, W)
        imgs_tensor = torch.from_numpy(np.array(images)).permute(0, 3, 1, 2).to(device)
        
        fid.update(imgs_tensor, real=True)
        
        # compute the statistics
        mu = (fid.real_features_sum / fid.real_features_num_samples).cpu().numpy()
        num_samples = fid.real_features_num_samples.item()
        features_cov_sum = fid.real_features_cov_sum.cpu().numpy()
        sigma = (features_cov_sum / num_samples) - np.outer(mu, mu)
        
        return mu, sigma

    def calculate_fid_from_stats(self, mu1, sigma1, mu2, sigma2):
        """
        numpy-based FID computation function (with numerical-stability corrections)
        mu1, sigma1: real-image statistics (Real)
        mu2, sigma2: generated-image statistics (Fake)
        eps: a very small constant for numerical stability
        """
        # 1. ensure the input data is converted to numpy arrays (guarantee 1-D/2-D)
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        eps = 1e-6
        
        #from scipy import linalg
        # 2. squared difference between the means
        diff = mu1 - mu2
       
        # 3. [KEY] add an epsilon diagonal to the covariance matrix for numerical stability
        # prevent the matrix from becoming singular and thus avoid sqrtm errors
        offset = np.eye(sigma1.shape[0]) * eps
        
        # 4. compute the geometric mean of the covariance matrices: sqrt(sigma1 * sigma2)
        # compensate with eps for errors that may occur in the sigma1.dot(sigma2) operation
        covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
        
        #1. matrix multiplication
        #cov_dot = sigma1.dot(sigma2)
        #2. compute the matrix square root
        #covmean = linalg.sqrtm(cov_dot)
       
        # 5. [IMPORTANT] remove tiny complex (imaginary) parts that appear during computation
        # tiny imaginary parts can appear due to numerical error; convert them to real values
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # 6. apply the final FID formula
        # d^2 = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        tr_covmean = np.trace(covmean)
        fid_score = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

        #4. apply the FID formula: ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        #fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        
        return fid_score

    def run_startup_test(self):
        """function that runs a one-time test inference at server startup and saves the result"""
        print("\n" + "="*60)
        print("[TEST] starting the initial inference and saving test at server startup...")
        test_prompt = "A high quality photo of a cat"
        
        try:
            # call ExecuteInference to run and save the file
            saved_paths = self.ExecuteInference(
                prompt=test_prompt,
                steps=20,     # it is only for testing, so reduce steps for a faster run
                cfg=7.5,
                batch_size=1,
                seed=42,
                output_path=OUTPUT_DIR, #/ "test_run" # test-only subfolder
                #sampler_type=getattr(request, 'SamplerType', 1) # assumes the proto has a SamplerType field
                sampler_type=1 #hardcoded
            )
            
            print(f"[TEST] initial inference and file saving successful!")
            for path in saved_paths:
                print(f" >> verified file: {path}")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"[TEST] initial inference test failed: {e}")

    def Heartbeat(self, request, context):
        """
        implement Heartbeat for server status checks
        """
        status_code = infaas_request_status_pb2.SUCCESS if self.pipe else infaas_request_status_pb2.UNAVAILABLE
        status_msg = infaas_request_status_pb2.InfaasRequestStatus(status=status_code)
        return query_pb2.HeartbeatResponse(status=status_msg)
    
async def serve():

    # 1. parse command-line arguments (integrating the process_option.py logic)
    parser = argparse.ArgumentParser(description="Diffusion gRPC Server")
    parser.add_argument("-model", type=int, required=True, help="Model ID (from models.json)")
    parser.add_argument("-sampler", type=int, required=True, help="sampler ID (from samplers.json)")
    parser.add_argument("-thread", type=int, default=10, help="thread number")
    parser.add_argument("-port", type=str, default="50051", help="Port number")
    parser.add_argument("-npz", type=str, required=True, help=".npz file path")    
    # additional arguments such as sampler or npz can be defined if needed
    args = parser.parse_args()

    # 2. determine the model filename
    models_data = load_json("models.json")
    if models_data is None:
        print("[FATAL ERROR] failed to load models.json")
        return

    model_filename = get_filename_by_id(models_data, args.model)
    if not model_filename:
        print(f"[FATAL ERROR] no model filename for ID {args.model}.")
        return

    # 3. determine the sampler
    samplers_data = load_json("samplers.json")
    if samplers_data is None: # failed to load the sampler data 확인
        print("[FATAL ERROR] failed to load samplers.json")
        return

    sampler_name = get_filename_by_id(samplers_data, args.sampler) # map the sampler name
    if not sampler_name: # check whether the ID exists
        print(f"[FATAL ERROR] no sampler name for ID {args.sampler}.")
        return

    print(f"[SYSTEM] starting to load model ID {args.model}({model_filename})...")
    print(f"[SYSTEM] starting to load sampler ID {args.model}({model_filename})...") 

    # 4. initialize ModelExecutor (passing all parameters)
    executor = ModelExecutor(
        model_id=args.model,
        model_filename=model_filename,
        sampler_id=args.sampler,
        sampler_name=sampler_name,
        thread_count=args.thread,
        port=args.port,        # pass the port number
        npz_path=args.npz
    )

    # check whether the model loaded successfully (exception handling)
    if executor.pipe is None:
        print("[FATAL ERROR] model loading failed, so the server cannot start.")
        return

    #server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server = grpc.aio.server()
    query_pb2_grpc.add_QueryServicer_to_server(executor, server)

    # bind the port using args.port
    server.add_insecure_port(f'[::]:{args.port}')
    #port = "50051"   #"8080"
    #actual_port = server.add_insecure_port(f'[::]:{port}')
    #actual_port = server.add_insecure_port(f'0.0.0.0:{port}')
    if args.port == 0:
        print(f"[ERROR] failed to bind port {port}! it may already be in use or you lack permission.")
        return  
    print(f"[SERVER] gRPC server started (Port: {args.port})")
    
    await server.start()
    await server.wait_for_termination()

# temporarily disabled.
if __name__ == "__main__":
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        print("shutting down the server...")
