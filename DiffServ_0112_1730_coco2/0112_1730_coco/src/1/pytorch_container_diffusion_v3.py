import grpc
from concurrent import futures
import torch
import io
from pathlib import Path

# 컴파일된 파일들을 루트에서 임포트
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

import argparse  # 추가
import json      # 추가

# ===============================================
# 🌟 전역 상수 설정 (INFaaS 폴더 구조 반영) 🌟
# ===============================================
MODEL_DIRECTORY = '/tmp/model'
OUTPUT_ROOT_DIR = '/tmp/diffusion_output' 
MODEL_FILE_NAME = "v1-5-pruned-emaonly.safetensors"
CKPT_FILE_PATH = Path(MODEL_DIRECTORY) / MODEL_FILE_NAME
OUTPUT_DIR = Path(OUTPUT_ROOT_DIR)

# 모델 파이프라인의 타입 정의 (명시적인 타입 힌트를 위해)
DiffusionPipeline = StableDiffusionPipeline

# [추가] JSON 로드 및 매핑 유틸리티
def load_json(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} 파일이 존재하지 않습니다.")
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
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
        print(f"[인포] ModelExecutor 초기화 (Model ID: {model_id}, File: {model_filename})")
        self.model_id = model_id # 모델 ID 저장
        self.model_directory = '/tmp/model'
        self.model_file_name = model_filename # 전달받은 파일명 사용
        self.ckpt_file_path = Path(self.model_directory) / self.model_file_name
        # 파라미터 저장
        self.sampler_id = sampler_id
        self.sampler_name = sampler_name
        self.thread_count = thread_count
        self.port = port     # 클래스 멤버 변수로 저장
        self.real_stats_path = npz_path

        # 모델 로드
        self.pipe, self.device = self.load_model()
        
        if self.pipe is None:
            print("[오류] 모델 로드 실패")
        # 2. 서버 시작 시 테스트 추론 1회 실행 (Warm-up)
        #if self.pipe:
        #    self.run_startup_test()
        #else:
        #    print("[오류] 모델 로드 실패로 테스트 추론을 건너뜁니다.")
        
        real_stats_path = "/workspace/real_stats.npz"
        if os.path.exists(real_stats_path):
            # load시 실사 이미지의 mu, sigma 값을 계산한다. 
            real_stats = np.load(real_stats_path)
            self.mu_real = real_stats['mu']
            self.sigma_real = real_stats['sigma']
            print(f"📊 실사 이미지의 mu 값 ({self.mu_real})")
            print(f"📊 실사 이미지의 sigma 값 계산 ({self.sigma_real})")

    def load_model(self) -> Tuple[Optional[StableDiffusionPipeline], str]:
        """
        단일 파일(.safetensors)로부터 Stable Diffusion 모델을 로드
        """

        total_start = time.perf_counter()
        # 1. 파일 시스템 로드 측정
        print(f"[타이머] 1단계: 가중치 파일 로딩 시작... ({self.ckpt_file_path})")
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
            print(f"[인포] 모델 로드 시도: {self.ckpt_file_path}")
            # from_single_file 메서드를 사용하여 체크포인트 로드
            pipe = StableDiffusionPipeline.from_single_file(
                str(self.ckpt_file_path),
                torch_dtype=dtype,
                load_safety_checker=False,
                low_cpu_mem_usage=False, # <--- 이 옵션을 추가하여 지연 로딩 방지
            )
            pipe.to(device)
            load_end = time.perf_counter()
            print(f" ✅ [완료] 파일 로드 시간: {load_end - load_start:.2f}초")
            

            # 2. 예열(Warm-up) 시간 측정
            # 여기서 model.safetensors 진행바가 다시 뜬다면 이 구간이 병목입니다.
            print(f"[타이머] 2단계: 컴포넌트 최적화(Warm-up) 시작...")
            warmup_start = time.perf_counter()

            with torch.inference_mode():
                # 안전검사기 완전 무시 (NSFW 오류 방지)
                pipe.safety_checker = None
                pipe(
                    prompt="a silent dog",
                    num_inference_steps=1,
                    width=128,
                    height=128,
                    guidance_scale=1.0
                )
            
            warmup_end = time.perf_counter()
            print(f" ✅ [완료] Warm-up 실행 시간: {warmup_end - warmup_start:.2f}초")

            total_end = time.perf_counter()
            print("=" * 60)
            print(f"🚀 [최종] 서버 준비 완료 총 소요 시간: {total_end - total_start:.2f}초")
            print("=" * 60)

            print(f"[인포] 가중치 로딩 100% 완료 ! GPU 준비됨")
            return pipe, device
        except Exception as e:
            print(f"[오류] 모델 로드 중 예외 발생: {e}")
            return None, device
    
    def QueryOnline(self, request, context):
        """
        gRPC QueryOnline 요청 처리
        """
        if self.pipe is None:
            status_msg = infaas_request_status_pb2.InfaasRequestStatus(
                status=infaas_request_status_pb2.UNAVAILABLE,
                msg="Model not loaded"
            )
            return query_pb2.QueryOnlineResponse(status=status_msg)

        print(f"[인포] 요청 수신 - 프롬프트: {request.Prompt}")

        # 추론 실행
        try:
            # --- 시간 측정 시작 ---
        # GPU 연산이 이전에 남아있을 수 있으므로 동기화 후 CPU 시간 기록
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            saved_paths = self.ExecuteInference(
                prompt=list(request.Prompt),
                steps=request.Steps if request.Steps > 0 else 30,
                cfg=request.CFG_Scale if request.CFG_Scale > 0 else 7.5,
                batch_size=request.BatchSize if request.BatchSize > 0 else 1,
                seed=request.Seed,
                #sampler_type=getattr(request, 'SamplerType', 1) # proto에 SamplerType 필드가 있다고 가정
                sampler_type=1 #하드코딩
                #height_px=256, 
                #width_px=256
            )

            # GPU 연산이 완전히 끝날 때까지 대기 후 종료 시간 기록
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            # --- 시간 측정 종료 ---

            elapsed_time = end_time - start_time
            print(f"------------------------------------------------------------")
            print(f"[성능 로그] 총 추론 시간: {elapsed_time:.4f} 초 (CPU+GPU)")
            print(f"------------------------------------------------------------")

            # 2. 결과 로그 출력
            print(f"[정보] 추론 및 파일 저장 완료!")
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
            return query_pb2.QueryOnlineResponse(
                image_paths=str_image_paths,
                status=status_msg
            )

        except Exception as e:
            print(f"[오류] 추론 중 오류 발생: {e}")
            status_msg = infaas_request_status_pb2.InfaasRequestStatus(
                status=infaas_request_status_pb2.INVALID,
                msg=str(e)
            )
            return query_pb2.QueryOnlineResponse(status=status_msg)

    def QueryOnlineImage(self, request, context):
        """
        gRPC QueryOnline 요청 처리
        """
        if self.pipe is None:
            status_msg = infaas_request_status_pb2.InfaasRequestStatus(
                status=infaas_request_status_pb2.UNAVAILABLE,
                msg="Model not loaded"
            )
            return query_pb2.QueryOnlineImageResponse(status=status_msg)

        print(f"[인포] 요청 수신 - 프롬프트: {request.Prompt}")
        print(f"[인포] request.Steps: {request.Steps}")
        print(f"[인포] request.Sampler_Type: {request.Sampler_Type}")

        # 추론 실행
        try:
            # --- 시간 측정 시작 ---
            # GPU 연산이 이전에 남아있을 수 있으므로 동기화 후 CPU 시간 기록
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            # 디렉토리명 생성
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            random_str = secrets.token_hex(3) # 6자리 랜덤 문자열
            unique_name = f"{now}_{random_str}"           
            target_dir = OUTPUT_DIR / unique_name
            print(target_dir)

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

            # GPU 연산이 완전히 끝날 때까지 대기 후 종료 시간 기록
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            # --- 시간 측정 종료 ---

            # --- npz파일 저장 ---
            mu_gen, sigma_gen = self.get_statistics(target_dir, self.device)
            # 2. 파일명 및 경로 설정 (target_dir 폴더 안에 gen_stats.npz로 명명) 
            out_file = os.path.join(target_dir, "gen_stats.npz")
            np.savez(out_file, mu=mu_gen, sigma=sigma_gen)
            print(f"✅ 통계 저장 완료: {out_file}")

            # --- [추가] FID 계산 로직 ---
            # 실사 이미지 통계 파일 경로
            # 1. 도커 내부의 절대 경로를 직접 지정 (가장 확실함)
            real_stats_path = "/workspace/real_stats.npz"
            fid_score = None

            if os.path.exists(real_stats_path):
                #print(f"📊 FID 계산 시작 (기준: {real_stats_path})")
                # 1. 실사 이미지 통계 로드
                #real_stats = np.load(real_stats_path)
                #mu_real = real_stats['mu']
                #sigma_real = real_stats['sigma']

                # 2. 제공된 함수를 호출하여 FID 계산
                fid_score = self.calculate_fid_from_stats(self.mu_real, self.sigma_real, mu_gen, sigma_gen)
                print(f"🔥 최종 FID Score: {fid_score:.4f}")
            else:
                print(f"⚠️ 경고: {real_stats_path} 파일을 찾을 수 없어 FID 계산을 건너뜁니다.")

            elapsed_time = end_time - start_time
            print(f"------------------------------------------------------------")
            print(f"[성능 로그] 총 추론 시간: {elapsed_time:.4f} 초 (CPU+GPU)")
            if fid_score is not None:
                print(f"[성능 로그] FID 점수: {fid_score:.4f}")
            print(f"------------------------------------------------------------")

            # 2. 결과 로그 출력
            print(f"[정보] 추론 및 파일 저장 완료!")
            str_image_paths = []
            raw_images_bytes = []
            for path in saved_paths:
                print(f" >> 저장된 위치: {path}")
                # Path 객체를 문자열로 변환하여 리스트에 추가
                str_image_paths.append(str(path))
                
                # 파일을 바이너리 읽기 모드('rb')로 열어 바이트 데이터 추출
                with open(path, "rb") as f:
                    raw_images_bytes.append(f.read())
            
            # 3. 성공 상태 메시지 구성
            status_msg = infaas_request_status_pb2.InfaasRequestStatus(
                status=infaas_request_status_pb2.SUCCESS
            )

            # 4. 수정된 proto 정의에 맞춰 image_paths 필드에 결과 담아 반환
            return query_pb2.QueryOnlineImageResponse(
                image_paths=str_image_paths,
                image_data=raw_images_bytes,  # 수정된 필드
                status=status_msg
            )

        except Exception as e:
            print(f"[오류] 추론 중 오류 발생: {e}")
            status_msg = infaas_request_status_pb2.InfaasRequestStatus(
                status=infaas_request_status_pb2.INVALID,
                msg=str(e)
            )
            return query_pb2.QueryOnlineImageResponse(status=status_msg)
        
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
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
            sampler_name = "Euler_a"
        elif sampler_type == 2:
            # DPM++ 2M Karras (매우 빠르고 고품질)
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
            print(f"[경고] 알 수 없는 샘플러 타입({sampler_type}), 기본값(Euler a)을 사용합니다.")
            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
            sampler_name = "Euler_a_Default"
        
        # 2. 난수 생성기 설정 (재현성 확보)
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
        output = self.pipe(
            prompt=prompt, # 리스트 형태의 프롬프트 전달
            num_inference_steps=steps,
            guidance_scale=cfg,
            height=height_px,
            width=width_px,
            generator=generator
        )
        
        generated_images = output.images

        # 5. 출력 폴더 생성 및 이미지 저장 (generate_and_save_image 로직 적용)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"생성된 이미지를 '{output_path}' 폴더에 저장합니다.")

        # 5. 이미지 저장 로직
        output_path.mkdir(parents=True, exist_ok=True)
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
       
        # 1. 행렬 곱셈
        cov_dot = sigma1.dot(sigma2)
       
        # 2. 행렬 제곱근 계산
        covmean = linalg.sqrtm(cov_dot)
       
        # 3. 수치적 불안정성으로 인한 허수 제거
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # 4. FID 공식 적용: ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

    def run_startup_test(self):
        """서버 시작 시 1회 테스트 추론을 실행하고 결과를 저장하는 함수"""
        print("\n" + "="*60)
        print("[테스트] 서버 기동 시 초기 추론 및 저장 테스트를 시작합니다...")
        test_prompt = "A high quality photo of a cat"
        
        try:
            # ExecuteInference를 호출하여 실행 및 파일 저장
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
            
            print(f"[테스트] 초기 추론 및 파일 저장 성공!")
            for path in saved_paths:
                print(f" >> 확인된 파일: {path}")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"[테스트] 초기 추론 테스트 실패: {e}")

    def Heartbeat(self, request, context):
        """
        서버 상태 확인을 위한 Heartbeat 구현
        """
        status_code = infaas_request_status_pb2.SUCCESS if self.pipe else infaas_request_status_pb2.UNAVAILABLE
        status_msg = infaas_request_status_pb2.InfaasRequestStatus(status=status_code)
        return query_pb2.HeartbeatResponse(status=status_msg)
    
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
    models_data = load_json("models.json")
    if models_data is None:
        print("[치명적 오류] models.json 로드 실패")
        return

    model_filename = get_filename_by_id(models_data, args.model)
    if not model_filename:
        print(f"[치명적 오류] ID {args.model}에 해당하는 모델 파일명이 없습니다.")
        return

    # 3. sampler 결정
    samplers_data = load_json("samplers.json")
    if samplers_data is None: # 샘플러 데이터 로드 실패 확인
        print("[치명적 오류] samplers.json 로드 실패")
        return

    sampler_name = get_filename_by_id(samplers_data, args.sampler) # 샘플러 이름 매핑
    if not sampler_name: # ID 존재 여부 확인
        print(f"[치명적 오류] ID {args.sampler}에 해당하는 샘플러 이름이 없습니다.")
        return

    print(f"[시스템] 모델 ID {args.model}({model_filename}) 로딩 시작...")
    print(f"[시스템] Sampler ID {args.model}({model_filename}) 로딩 시작...") 

    # 4. ModelExecutor 초기화 (모든 파라미터 전달)
    executor = ModelExecutor(
        model_id=args.model,
        model_filename=model_filename,
        sampler_id=args.sampler,
        sampler_name=sampler_name,
        thread_count=args.thread,
        port=args.port,        # 포트 번호 전달
        npz_path=args.npz
    )

    # 모델 로드 성공 여부 확인 (예외 처리)
    if executor.pipe is None:
        print("[치명적 오류] 모델 로드에 실패하여 서버를 시작할 수 없습니다.")
        return

    #server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.thread))
    query_pb2_grpc.add_QueryServicer_to_server(executor, server)

    # args.port를 사용하여 포트 바인딩
    server.add_insecure_port(f'[::]:{args.port}')
    #port = "50051"   #"8080"
    #actual_port = server.add_insecure_port(f'[::]:{port}')
    #actual_port = server.add_insecure_port(f'0.0.0.0:{port}')
    if args.port == 0:
        print(f"[오류] 포트 {port} 바인딩에 실패했습니다! 이미 사용 중이거나 권한이 없습니다.")
        return  
    print(f"[서버] gRPC 서버 시작됨 (Port: {args.port})")
    server.start()
    server.wait_for_termination()

# 임시로 막는다.
if __name__ == "__main__":
    serve()