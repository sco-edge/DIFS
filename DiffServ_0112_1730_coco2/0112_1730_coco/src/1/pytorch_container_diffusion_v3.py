# Purpose: Worker Server; for gRPC Worker process
import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__),"..")
    )

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


import traceback
import time
from typing import Optional, Tuple
from pathlib import Path
from concurrent import futures
import json
import datetime
import secrets
import argparse
import torch
import grpc
import numpy as np
from PIL import Image
from scipy import linalg


# Import compiled files from the root directory
from project_paths import * 
import config


#from experiment_models import MODELS
#executor.preload_models(MODELS)

from worker_metrics import WorkerMetrics
from torchmetrics.image.fid import FrechetInceptionDistance

import query_pb2
import query_pb2_grpc
import infaas_request_status_pb2

### Merge
from diffusers import StableDiffusionPipeline

import diffusers
import transformers

print("TORCH=", torch.__version__)
print("DIFFUSERS=", diffusers.__version__)
print("TRANSFORMERS=", transformers.__version__)



from diffusers import (
    EulerAncestralDiscreteScheduler, 
    DPMSolverMultistepScheduler, 
    LMSDiscreteScheduler, 
    DDIMScheduler
)

from project_paths import (
    PROJECT_ROOT,
    MODELS_JSON,
    SAMPLERS_JSON,
    REAL_STATS,
    MODEL_CACHE,
    DIFFUSION_OUTPUT,
    OUTPUT_DIR,
    METRICS_DIR
)

# ========================================================================
# 🌟 Global Constant Settings (Reflecting the INFaaS Folder Structure)  🌟
# ========================================================================
MODEL_DIR = MODEL_CACHE

#OUTPUT_ROOT_DIR = '/tmp/diffusion_output' 
OUTPUT_DIR = DIFFUSION_OUTPUT

MODEL_FILE_NAME = "sd-v1-4.safetensors"
#MODEL_FILE_NAME = "v1-5-pruned-emaonly.safetensors"
CKPT_FILE_PATH = Path(MODEL_DIR) / MODEL_FILE_NAME
#OUTPUT_DIR = Path(OUTPUT_ROOT_DIR)

# [추가] JSON 로드 및 매핑 유틸리티
def load_json(file_path):

    file_path = Path(file_path)

    print("=" * 60)
    print("[JSON LOADER]")
    print("Trying:", file_path)
    print("=" * 60)

    if not file_path.exists():
        print(f"Error: {file_path} does not exist.")
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_filename_by_id(data, target_id):
    for item in data:
        if item.get("id") == target_id:
            return item.get("filename")
    return None

class ModelExecutor(query_pb2_grpc.QueryServicer):
    #def __init__(self, model_id: int, model_filename: str): # 인자 추가
    def __init__(
        self, 
        model_id: int, 
        model_filename: str, 
        sampler_id: int, 
        sampler_name: str, 
        thread_count: int, 
        port: int,           # 포트 번호 추가
        npz_path: str
    ):
        """
        ModelExecutor 초기화 및 모델 로드
        """        
        print(f"[Info] Initializing ModelExecutor (Model ID: {model_id}, File: {model_filename})")
        self.model_id = model_id # 모델 ID 저장
        self.model_directory = MODEL_DIR
        self.model_file_name = model_filename # 전달받은 파일명 사용
        self.ckpt_file_path = Path(self.model_directory) / self.model_file_name
        # 파라미터 저장
        self.sampler_id = sampler_id
        self.sampler_name = sampler_name
        self.thread_count = thread_count
        self.port = port     # 클래스 멤버 변수로 저장
        self.local_queue_length = 0
        ###########################################################
        # Worker runtime state
        ###########################################################
        self.busy = False
        self.active_requests = 0

        ###########################################################
        # Dynamic model switching
        ###########################################################
        import threading

        self.model_lock = threading.Lock()

        self.current_model_name = model_filename

        self.model_switches = 0
        self.last_switch_time_ms = 0.0

        ###########################################################
        # Metrics logger
        ###########################################################
        self.metrics = WorkerMetrics(output_dir=str(METRICS_DIR))
        print("=" * 60)
        print(f"[WORKER] Metrics directory: {self.metrics.output_dir}")
        print("=" * 60)

        self.real_stats_path = npz_path

        self.mu_real = None
        self.sigma_real = None

        # 모델 로드
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        #####################################################
        # Multiple loaded pipelines
        #####################################################

        self.model_cache = {}
        self.current_pipe = None
        
        #####################################################
        #####################################################
        self.current_pipe = self.load_model(self.model_file_name)

        if self.current_pipe is None:
            raise RuntimeError(
                f"Failed to load model: {self.model_file_name}"
            )

        self.model_cache[self.model_file_name] = self.current_pipe
        #####################################################
        # 2. 서버 시작 시 테스트 추론 1회 실행 (Warm-up)
        #if self.pipe:
        #    self.run_startup_test()
        #else:
        #    print("[오류] 모델 로드 실패로 테스트 추론을 건너뜁니다.")
        
        DEFAULT_STATS = REAL_STATS

        real_stats_path = self.real_stats_path
        if os.path.exists(self.real_stats_path):
            # load시 실사 이미지의 mu, sigma 값을 계산한다. 
            real_stats = np.load(real_stats_path)
            self.mu_real = real_stats['mu']
            self.sigma_real = real_stats['sigma']
            print(f"📊 The mu value of a real-world image ({self.mu_real})")
            print(f"📊 Calculating the sigma value of a real-world image ({self.sigma_real})")
        elif os.path.exists(DEFAULT_STATS ):
            self.real_stats_path = DEFAULT_STATS 
            print(f"npz path not picked from command line, using the hard coded path: {DEFAULT_STATS }")
            real_stats = np.load(DEFAULT_STATS )
            self.mu_real = real_stats['mu']
            self.sigma_real = real_stats['sigma']
            print(f"📊 The mu value of a real-world image ({self.mu_real})")
            print(f"📊 Calculating the sigma value of a real-world image ({self.sigma_real})")


    def load_model(self, model_filename=None) -> Tuple[Optional[StableDiffusionPipeline], str]:
        """
        단일 파일(.safetensors)로부터 Stable Diffusion 모델을 로드
        """

        total_start = time.perf_counter()
        # 1. 파일 시스템 로드 측정
        print(f"[Timer] Step 1: Starting to load the weight file... ({self.ckpt_file_path})")
        load_start = time.perf_counter()

        ############################################################
        # GPU MEMORY BEFORE LOADING
        ############################################################        
        
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()

            print("=" * 60)
            print("[GPU BEFORE LOAD]")
            print(f"Free : {free / 1024**2:.2f} MB")
            print(f"Total: {total / 1024**2:.2f} MB")
            print("=" * 60)
            
            device = "cuda"
            dtype = torch.float16
            print(f"cuda")
        else:
            device = "cpu"
            dtype = torch.float32
            print(f"cpu")

        try:
            if model_filename is not None:

                self.model_file_name = model_filename
                ckpt_file_path = Path(self.model_directory) / model_filename
            print(f"[MODEL] Loading {ckpt_file_path}")

            print("=" * 60)
            print("CHECKPOINT PATH =", ckpt_file_path)
            print("EXISTS =", os.path.exists(ckpt_file_path))

            if os.path.exists(MODEL_DIR):
                print(f"CONTENTS OF {MODEL_DIR}:")
                print(os.listdir(MODEL_DIR))
            else:
                print(f"{MODEL_DIR} DOES NOT EXIST")

            print("=" * 60)

            if ckpt_file_path.is_dir():
    
                print("[MODEL] Loading HuggingFace directory")

                pipe = StableDiffusionPipeline.from_pretrained(
                    str(ckpt_file_path),
                    torch_dtype=dtype,
                    safety_checker=None,
                    local_files_only=True,
                )

            else:

                print("[MODEL] Loading single checkpoint")

                pipe = StableDiffusionPipeline.from_single_file(
                    str(ckpt_file_path),
                    torch_dtype=torch.float16,
                    load_safety_checker=False,
                    low_cpu_mem_usage=True,
                )

                ############################################################
                # Memory Optimizations
                ############################################################

                pipe.enable_attention_slicing()
                pipe.enable_vae_slicing()
                pipe.enable_model_cpu_offload()

                print("=" * 60)
                print("[MODEL] Attention slicing enabled")
                print("[MODEL] VAE slicing enabled")
                print("[MODEL] CPU offload enabled")
                print("=" * 60)

            print("ROOT DIRECTORY")
            print(os.listdir("/"))

            print("TMP DIRECTORY")
            print(os.listdir("/tmp"))

            for root, dirs, files in os.walk("/tmp"):
                for file in files:
                    if file.endswith(".safetensors"):
                        print("FOUND SAFETENSOR:", os.path.join(root, file))

            # pipe.to(device)
            load_end = time.perf_counter()
            print(f" ✅ [Complete] File load time: {load_end - load_start:.2f}초")
            

            # 2. 예열(Warm-up) 시간 측정
            # 여기서 model.safetensors 진행바가 다시 뜬다면 이 구간이 병목입니다.
            print("[Timer] Step 2: Component Optimization (Warm-up) Begins...")
            warmup_start = time.perf_counter()

            with torch.inference_mode():
                # 안전검사기 완전 무시 (NSFW 오류 방지)
                pipe.safety_checker = None
                pipe(
                    prompt="dog",
                    num_inference_steps=1,
                    width=64,
                    height=64,
                    guidance_scale=1.0
                )
            
            warmup_end = time.perf_counter()
            print(f" ✅ [Completed] Warm-up execution time: {warmup_end - warmup_start:.2f}초")
            ############################################################
            # GPU MEMORY AFTER LOADING
            ############################################################
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()

                print("=" * 60)
                print("[GPU AFTER LOAD]")
                print(f"Free : {free / 1024**2:.2f} MB")
                print(f"Total: {total / 1024**2:.2f} MB")
                print("=" * 60)

            total_end = time.perf_counter()

            self.metrics.log_model_load(
                model=self.model_file_name,
                load_time=total_end - total_start
            )

            print("=" * 60)
            print(f"🚀 [Final] Server setup complete. Total time taken: {total_end - total_start:.2f}초")
            print("=" * 60)

            print("[Info] Weight loading 100% complete! GPU ready")

            ############################################################
            # Release any unused cached GPU memory
            ############################################################
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return pipe
        except Exception as e:
            print("[Error] An exception occurred while loading the model.")
            print(e)
            traceback.print_exc()
            return None

    def replace_model(self, model_filename):
    
        start = time.perf_counter()
        print(f"[MODEL SWITCH] {self.current_model_name} -> {model_filename}")

        with self.model_lock:

            if model_filename in self.model_cache:
                self.current_pipe = self.model_cache[model_filename]

            else:
                self.current_pipe = self.load_model(model_filename)

                if self.current_pipe is None:

                    raise RuntimeError(
                        f"Cannot load {model_filename}"
                    )

                self.model_cache[model_filename] = self.current_pipe

            self.current_model_name = model_filename
            self.model_file_name = model_filename

            self.model_switches += 1

        elapsed = (time.perf_counter() - start) * 1000

        self.last_switch_time_ms = elapsed
        print(f"[MODEL SWITCH] completed in {elapsed:.2f} ms")
        return elapsed

    def hot_switch_model(self,
                        model_name):

        start = time.perf_counter()

        with self.model_lock:

            self.current_pipe = self.model_cache[
                model_name
            ]

            self.current_model_name = model_name

            self.model_switches += 1

        elapsed = (time.perf_counter()-start)*1000

        self.last_switch_time_ms = elapsed

        return elapsed
    
    def QueryOnline(self, request, context):
        """
        gRPC QueryOnline 요청 처리
        """
        if self.current_pipe is None:
            status_msg = infaas_request_status_pb2.InfaasRequestStatus(
                status=infaas_request_status_pb2.UNAVAILABLE,
                msg="Model not loaded"
            )
            return query_pb2.QueryOnlineResponse(status=status_msg)

        print(f"[Info] Request Received - Prompt: {request.Prompt}")

        # 추론 실행
        try:
            # --- 시간 측정 시작 ---
        # GPU 연산이 이전에 남아있을 수 있으므로 동기화 후 CPU 시간 기록
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.active_requests += 1
            self.busy = self.active_requests > 0
            start_time = time.perf_counter()

            with self.model_lock: # protect inference
                saved_paths = self.ExecuteInference(
                    prompt=list(request.Prompt),
                    steps=request.Steps if request.Steps > 0 else 30,
                    cfg=request.CFG_Scale if request.CFG_Scale > 0 else 7.5,
                    batch_size=request.BatchSize if request.BatchSize > 0 else 1,
                    seed=request.Seed,
                    #sampler_type=1 #하드코딩
                    sampler_type=self.sampler_id
                    #height_px=256, 
                    #width_px=256
                )

            # GPU 연산이 완전히 끝날 때까지 대기 후 종료 시간 기록
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            # --- 시간 측정 종료 ---

            elapsed_time = end_time - start_time

            # QueryOnline() does not compute FID
            fid_value = -1
            self.metrics.log_inference(
                worker_port=self.port,
                model_id=self.model_id,
                model_name=self.current_model_name,
                sampler=self.sampler_name,
                batch_size=request.BatchSize,
                steps=request.Steps,
                latency=elapsed_time,
                fid=fid_value,
                images_generated=len(saved_paths),
                success=True
            )
            print("------------------------------------------------------------")
            print(f"[Performance Log] Total inference time: {elapsed_time:.4f} seconds (CPU+GPU)")
            print("------------------------------------------------------------")

            # 2. 결과 로그 출력
            print("[정보] 추론 및 파일 저장 완료!")
            str_image_paths = []
            for path in saved_paths:
                print(f" >> 저장된 위치: {path}")
                # Path 객체를 문자열로 변환하여 리스트에 추가
                str_image_paths.append(str(path))
            
            # 3. 성공 상태 메시지 구성
            status_msg = infaas_request_status_pb2.InfaasRequestStatus(
                status=infaas_request_status_pb2.SUCCESS
            )

            # 4. 수정된 proto 정의에 맞춰 image_paths 필드에 결과 담아 반환
            self.active_requests -= 1
            self.busy = self.active_requests > 0
            return query_pb2.QueryOnlineResponse(
                image_paths=str_image_paths,
                status=status_msg
            )

        except Exception as e:
            print("[Error] An error occurred during inference")
            print(e)
            traceback.print_exc()
            status_msg = infaas_request_status_pb2.InfaasRequestStatus(
                status=infaas_request_status_pb2.INVALID,
                msg="Inference failed"
            )

            self.busy = False
            if self.active_requests > 0:
                self.active_requests -= 1

            return query_pb2.QueryOnlineResponse(status=status_msg)
        finally:
            if self.active_requests > 0:
                self.active_requests -= 1

            self.busy = self.active_requests > 0


    def QueryOnlineImage(self, request, context):
        """
        gRPC QueryOnline 요청 처리
        """
        if self.current_pipe is None:
            status_msg = infaas_request_status_pb2.InfaasRequestStatus(
                status=infaas_request_status_pb2.UNAVAILABLE,
                msg="Model not loaded"
            )
            yield query_pb2.QueryOnlineImageResponse(status=status_msg)
            return

        print(f"[Info] Request Received - Prompt: {request.Prompt}")

        print("[DEBUG-1] entered QueryOnlineImage")

        print(f"[Info] request.Steps: {request.Steps}")
        print(f"[Info] request.Sampler_Type: {request.Sampler_Type}")

        # 추론 실행
        try:
            # --- Start Time ---
            # Since GPU computations may still be in progress, record CPU time after synchronization
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            self.active_requests += 1
            self.busy = self.active_requests > 0

            start_time = time.perf_counter()

            # Creating a Directory Name
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            random_str = secrets.token_hex(3) # 6자리 랜덤 문자열
            unique_name = f"{now}_{random_str}"           
            target_dir = OUTPUT_DIR / unique_name
            print("[DEBUG-2] output dir created")

            print(target_dir)

            with self.model_lock:
                saved_paths = self.ExecuteInference(
                    prompt=list(request.Prompt),
                    steps=request.Steps if request.Steps > 0 else 30,
                    cfg=request.CFG_Scale if request.CFG_Scale > 0 else 7.5,
                    batch_size=request.BatchSize if request.BatchSize > 0 else 1,
                    seed=request.Seed,
                    #sampler_type=getattr(request, 'SamplerType', 1) # proto에 SamplerType 필드가 있다고 가정
                    sampler_type=request.Sampler_Type, #하드코딩
                    #height_px=256, 
                    #width_px=256
                    output_path=target_dir
                )
                print("[DEBUG-3] ExecuteInference finished")
                print(saved_paths)

            # GPU 연산이 완전히 끝날 때까지 대기 후 종료 시간 기록
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            # --- 시간 측정 종료 ---

            # --- npz파일 저장 ---
            image_dir = Path(saved_paths[0]).parent
            mu_gen, sigma_gen = self.get_statistics(image_dir, self.device)
            print("[DEBUG-4] statistics computed")

            # 2. 파일명 및 경로 설정 (target_dir 폴더 안에 gen_stats.npz로 명명) 
            out_file = os.path.join(target_dir, "gen_stats.npz")
            np.savez(out_file, mu=mu_gen, sigma=sigma_gen)
            print(f"✅ 통계 저장 완료: {out_file}")

            # --- [추가] FID 계산 로직 ---
            # 실사 이미지 통계 파일 경로
            # 1. 도커 내부의 절대 경로를 직접 지정 (가장 확실함)
            
            fid_score = None

            if self.mu_real is not None:
                #print(f"📊 FID 계산 시작 (기준: {real_stats_path})")
                # 1. 실사 이미지 통계 로드
                #real_stats = np.load(real_stats_path)
                #mu_real = real_stats['mu']
                #sigma_real = real_stats['sigma']

                # 2. 제공된 함수를 호출하여 FID 계산
                fid_score = self.calculate_fid_from_stats(
                    self.mu_real, 
                    self.sigma_real,
                    mu_gen, 
                    sigma_gen)
                print("[DEBUG-5] fid computed")

                print(f"Final FID Score: {fid_score:.4f}")
                if hasattr(self.metrics, "log_fid"):
                    self.metrics.log_fid(fid_score)
            else:
                print("Skipping FID because no reference statistics are available.")

            elapsed_time = end_time - start_time


            # If FID could not be computed
            fid_value = fid_score if fid_score is not None else -1

            self.metrics.log_inference(
                worker_port=self.port,
                model_id=self.model_id,
                sampler=self.sampler_name,
                batch_size=request.BatchSize,
                steps=request.Steps,
                latency=elapsed_time,
                fid=fid_value,
                images_generated=len(saved_paths),
                success=True
            )
            print("------------------------------------------------------------")
            print(f"[Performance Log] Total inference time: {elapsed_time:.4f} seconds (CPU+GPU)")
            if fid_score is not None:
                print(f"[성능 로그] FID 점수: {fid_score:.4f}")
            print("------------------------------------------------------------")

                        
            # 3. 성공 상태 메시지 구성
            status_msg = infaas_request_status_pb2.InfaasRequestStatus(
                status=infaas_request_status_pb2.SUCCESS
            )

            print("[DEBUG-6] beginning image streaming")
            for path in saved_paths:
                print(f" >> 저장된 위치: {path}")

                print(path)
                with open(path, "rb") as f:
                    image_bytes = f.read()


                print("[DEBUG-7] yielding response")
                yield query_pb2.QueryOnlineImageResponse(
                    image_path=str(path),
                    image_data=image_bytes,
                    status=status_msg
                )

            self.active_requests -= 1
            self.busy = self.active_requests > 0

            return

        except Exception as e:
    
            print("="*80)
            print("WORKER EXCEPTION")
            print("="*80)
            traceback.print_exc()
            print(repr(e))
            print("="*80)

            print("[Error] An error occurred during inference")
            status_msg = infaas_request_status_pb2.InfaasRequestStatus(
                status=infaas_request_status_pb2.INVALID,
                msg="Inference failed"
            )

            self.busy = False
            if self.active_requests > 0:
                self.active_requests -= 1

            yield query_pb2.QueryOnlineImageResponse(
                status=status_msg
            )
            return
        
    def ExecuteInference(self, prompt: list, steps: int, cfg: float, batch_size: int, seed: int,
                         sampler_type: int = 1,  # 1: Euler a, 2: DPM++, 3: LMS, 4: DDIM 
                         height_px: int = 384, width_px: int = 384, output_path: Path = OUTPUT_DIR):
        """
        실제 Diffusion 추론 엔진 실행 및 이미지 저장
        """
        print(f"\n--- 이미지 추론 시작 (Sampler Type: {sampler_type} 및 저장 프로세스 시작) ---")
        
        # 1. 샘플러(Scheduler) 설정
        if sampler_type == 1:
            # Euler Ancestral (Euler a)
            self.current_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.current_pipe.scheduler.config)
            sampler_name = "Euler_a"
        elif sampler_type == 2:
            # DPM++ 2M Karras (매우 빠르고 고품질)
            self.current_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.current_pipe.scheduler.config, use_karras_sigmas=True
            )
            sampler_name = "DPM++_Karras"
        elif sampler_type == 3:
            # LMS (Linear Multi-Step)
            self.current_pipe.scheduler = LMSDiscreteScheduler.from_config(self.current_pipe.scheduler.config)
            sampler_name = "LMS"
        elif sampler_type == 4:
            # DDIM
            self.current_pipe.scheduler = DDIMScheduler.from_config(self.current_pipe.scheduler.config)
            sampler_name = "DDIM"
        else:
            print(f"[경고] 알 수 없는 샘플러 타입({sampler_type}), 기본값(Euler a)을 사용합니다.")
            self.current_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.current_pipe.scheduler.config)
            sampler_name = "Euler_a_Default"
        
        # 2. 난수 생성기 설정 (재현성 확보)
        if seed == 0:
                seed = torch.seed()

        generator = torch.Generator(self.device).manual_seed(seed)
        #prompt_list = [prompt] * batch_size

        # 3. 프롬프트 리스트 검증
        # 입력된 프롬프트 개수가 batch_size보다 적을 경우를 대비한 처리
        if len(prompt) < batch_size:
            print(f"[경고] 입력된 프롬프트({len(prompt)})가 BatchSize({batch_size})보다 적습니다. 부족분은 마지막 프롬프트로 채웁니다.")
            prompt = prompt + [prompt[-1]] * (batch_size - len(prompt))
        elif len(prompt) > batch_size:
            print(f"[정보] BatchSize({batch_size})에 맞춰 입력된 프롬프트 중 앞부분만 사용합니다.")
            prompt = prompt[:batch_size]

        # 4. 이미지 생성
        print(f"총 {batch_size}개의 이미지 생성 중... (Sampler: {sampler_name})")
        print(f"총 {batch_size}개의 이미지 ({width_px}x{height_px})")
        output = self.current_pipe(
            prompt=prompt, # 리스트 형태의 프롬프트 전달
            num_inference_steps=steps,
            guidance_scale=cfg,
            height=height_px,
            width=width_px,
            generator=generator
        )
        
        generated_images = output.images

        if torch.cuda.is_available():
    
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)

            reserved = torch.cuda.memory_reserved() / (1024 ** 2)

            self.metrics.log_gpu_memory(
                allocated_mb=allocated,
                reserved_mb=reserved
            )

       
        # 5. Create an output folder and save the image (apply the `generate_and_save_image` logic)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving generated images to Output Directory: ‘{output_path}’.")
        saved_files = []
        for i, image in enumerate(generated_images):
            # 파일명 형식: sd_output_[샘플러명]_[인덱스]_[시드]_[해상도].png
            filename = output_path / f"sd_{sampler_name}_b{i+1}_s{seed}_{width_px}x{height_px}.png"
            image.save(filename)
            print(f"  -> 저장 완료: {filename}")
            saved_files.append(filename)
            print(f" -> 저장 완료: {filename} (Prompt: {prompt[i][:20]}...) ")

        print("-" * 50)
        return saved_files # 저장된 파일 경로 리스트 반환

    # --- 1. 통계량(mu, sigma) 추출 함수 ---
    def get_statistics(self, image_dir, device):
        fid = FrechetInceptionDistance(feature=2048).to(device)
        
        valid_extensions = ('.png', '.jpg', '.jpeg')
        images = []
        files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)])
        
        if not files:
            raise ValueError(f"❌ 폴더에 이미지가 없습니다: {image_dir}")

        print(f"📦 {len(files)}개 이미지 로드 중...")
        for filename in files:
            img = Image.open(os.path.join(image_dir, filename)).convert('RGB')
            img = img.resize((299, 299), Image.LANCZOS)
            images.append(np.array(img))

        # (N, H, W, C) -> (N, C, H, W)
        imgs_tensor = torch.from_numpy(np.array(images)).permute(0, 3, 1, 2).to(device)
        
        fid.update(imgs_tensor, real=True)
        
        # 통계량 산출
        mu = (fid.real_features_sum / fid.real_features_num_samples).cpu().numpy()
        num_samples = fid.real_features_num_samples.item()
        features_cov_sum = fid.real_features_cov_sum.cpu().numpy()
        sigma = (features_cov_sum / num_samples) - np.outer(mu, mu)
        
        return mu, sigma

    def calculate_fid_from_stats(self, mu1, sigma1, mu2, sigma2):
        """
        두 분포의 평균(mu)과 공분산(sigma)을 사용하여 FID를 계산합니다.
        """
        #from scipy import linalg
       
        diff = mu1 - mu2
       
        # 1. Matrix Multiplication
        cov_dot = sigma1.dot(sigma2)
       
        # 2. Calculating the Square Root of a Matrix
        covmean, _ = linalg.sqrtm(cov_dot, disp=False)

        if not np.isfinite(covmean).all():
            eps = 1e-6
            covmean = linalg.sqrtm(
                (sigma1 + eps*np.eye(sigma1.shape[0])) @
                (sigma2 + eps*np.eye(sigma2.shape[0]))
            )
       
        # 3. Elimination of Imaginary Numbers Due to Numerical Instability
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # 4. Official Application of FID: ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

    def run_startup_test(self):
        
        # """서버 시작 시 1회 테스트 추론을 실행하고 결과를 저장하는 함수"""
        """A function that runs a single test inference when the server starts and saves the results"""

        print("\n" + "="*60)

        # print("[테스트] 서버 기동 시 초기 추론 및 저장 테스트를 시작합니다...")
        print("[Test] Starting the initial inference and storage test upon server startup...")

        # test_prompt = "A high quality photo of a cat"
        test_prompt = ["A high quality photo of a cat"]
        
        try:
            # ExecuteInference를 호출하여 실행 및 파일 저장
            if self.current_pipe is None:
                print("No model loaded.")
                return
            else:
                saved_paths = self.ExecuteInference(
                    prompt=test_prompt,
                    steps=20,     # 테스트용이므로 빠른 실행을 위해 steps 하향
                    cfg=7.5,
                    batch_size=1,
                    seed=42,
                    output_path=OUTPUT_DIR, #/ "test_run" # 테스트 전용 하위 폴더
                    #sampler_type=getattr(request, 'SamplerType', 1) # proto에 SamplerType 필드가 있다고 가정
                    sampler_type=1 #하드코딩
                )
            
            print("[테스트] 초기 추론 및 파일 저장 성공!")
            for path in saved_paths:
                print(f" >> 확인된 파일: {path}")
            print("="*60 + "\n")
            
        except Exception as e:
            print("[Test] Initial inference test failed")
            print(e)
            traceback.print_exc()

    def Heartbeat(self, request, context):
        """
        Implementing Heartbeat to Check Server Status
        """
        status_code = (infaas_request_status_pb2.SUCCESS if self.current_pipe is not None else infaas_request_status_pb2.UNAVAILABLE)
        status_msg = infaas_request_status_pb2.InfaasRequestStatus(status=status_code)
        self.metrics.record_heartbeat()
        return query_pb2.HeartbeatResponse(status=status_msg)


    def WorkerStatus(self, request, context):
    
        gpu_mem, gpu_reserved, gpu_peak = self.metrics.gpu_memory()

        return query_pb2.WorkerStatusResponse(
            workload_type=str(self.sampler_name),
            model_name=str(self.model_file_name),
            busy=self.busy,
            queue_length=self.local_queue_length,
            gpu_memory_mb=gpu_mem,
            average_latency_ms=self.metrics.average_latency() * 1000,
            active_requests=self.active_requests,
            worker_port=int(self.port),
            throughput=self.metrics.current_throughput(),
            gpu_reserved_mb=gpu_reserved,
            gpu_peak_mb=gpu_peak,
            last_fid=self.metrics.current_fid(),
            total_images_generated=self.metrics.total_images_generated(),
            model_switches=self.model_switches,
            last_switch_time_ms=self.last_switch_time_ms
        )

    def LoadModel(self, request, context):
        '''
        Implementing LoadModel RPC
        '''

        # elapsed = self.replace_model(request.model_name)
        if config.MODEL_SWITCH_MODE == "cold":

            elapsed = self.replace_model(request.model_name)
        else:
            elapsed = self.hot_switch_model(request.model_name)

        return query_pb2.LoadModelResponse(
            success=True,
            switch_time_ms=elapsed,
            current_model=request.model_name
        )

    def load_pipeline(self, model_filename):
    
        if model_filename in self.model_cache:

            return self.model_cache[model_filename]

        print(f"[CACHE] Loading {model_filename}")
        pipe = self.load_model(model_filename)

        if pipe is not None:

            self.model_cache[model_filename] = pipe

        return pipe

    def preload_models(self, model_list):
    
        for model in model_list:    
            filename = model["filename"]
            self.load_pipeline(filename)

        first = model_list[0]["filename"]

        self.current_pipe = self.model_cache[first]
        self.current_model_name = first


def serve():

    # 1. 명령행 인자 파싱 (process_option.py 로직 통합)
    parser = argparse.ArgumentParser(description="Diffusion gRPC Server")
    parser.add_argument("-model", type=int, required=True, help="Model ID (from models.json)")
    parser.add_argument("-sampler", type=int, required=True, help="sampler ID (from samplers.json)")
    parser.add_argument("-thread", type=int, default=10, help="thread number")
    parser.add_argument("-port", type=str, default="50051", help="Port number")
    parser.add_argument("-npz", type=str, required=True, help=".npz file path")    
    # 필요한 경우 sampler, npz 등 추가 인자 정의 가능
    args = parser.parse_args()

    # 2. 모델 파일명 결정
    models_data = load_json(str(MODELS_JSON))
    if models_data is None:
        print("[Fatal Error] models.json Failed to load")
        return

    # 3. sampler 결정
    samplers_data = load_json(str(SAMPLERS_JSON))
    if samplers_data is None: # 샘플러 데이터 Failed to load 확인
        print("[Fatal Error] samplers.json Failed to load")
        return

    ############################################################
    # Verify project paths
    ############################################################
    print("=" * 70)
    print("MODELS_JSON   :", MODELS_JSON)
    print("SAMPLERS_JSON :", SAMPLERS_JSON)
    print("=" * 70)

    model_filename = get_filename_by_id(models_data, args.model)
    if not model_filename:
        print(f"[Fatal Error] ID {args.model}에 해당하는 모델 파일명이 없습니다.")
        return

    sampler_name = get_filename_by_id(samplers_data, args.sampler) # 샘플러 이름 매핑
    if not sampler_name: # ID 존재 여부 확인
        print(f"[Fatal Error] ID {args.sampler}에 해당하는 샘플러 이름이 없습니다.")
        return

    print(f"[시스템] 모델 ID {args.model}({model_filename}) 로딩 시작...")
    print(f"[시스템] Sampler ID {args.sampler}({sampler_name}) 로딩 시작...") 

    # 4. ModelExecutor 초기화 (모든 파라미터 전달)
    executor = ModelExecutor(
        model_id=args.model,
        model_filename=model_filename,
        sampler_id=args.sampler,
        sampler_name=sampler_name,
        thread_count=args.thread,
        port=args.port, 
        npz_path=args.npz
    ) # initializing model executors


    # 모델 로드 성공 여부 확인 (예외 처리)
    # Check if the model loaded successfully (exception handling)
    if executor.current_pipe is None:
        #print("[Fatal Error] The server cannot be started because the model failed to load.")
        print("[Fatal Error] The server cannot start because the model failed to load.")
        return

    #server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.thread))
    query_pb2_grpc.add_QueryServicer_to_server(executor, server)

    # args.port를 사용하여 포트 바인딩
    actual_port = server.add_insecure_port(f"[::]:{args.port}")

    if actual_port == 0:
        print("Failed to bind port")
        return
        
    

    print("[DEBUG] before server.start()")
    server.start()
    #print(f"[서버] gRPC 서버 시작됨 (Port: {args.port})")
    print(f"[Server] gRPC server started (Port: {args.port})")

    print("[DEBUG] entering wait_for_termination")
    server.wait_for_termination()
    print("[DEBUG] returned from wait_for_termination")

# 임시로 막는다.
if __name__ == "__main__":
    
    try: # Catch the KeyboardInterrupt inside the worker; Avoiding the long traceback
        serve()

    except KeyboardInterrupt:
        print("\n[WORKER] Shutdown requested")

    finally:
        print("[WORKER] Exited cleanly")