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

# ===============================================
# 🌟 전역 상수 설정 (INFaaS 폴더 구조 반영) 🌟
# ===============================================
MODEL_DIRECTORY = '/tmp/model'
OUTPUT_ROOT_DIR = '/tmp/infaas_output' 
MODEL_FILE_NAME = "v1-5-pruned-emaonly.safetensors"
CKPT_FILE_PATH = Path(MODEL_DIRECTORY) / MODEL_FILE_NAME
OUTPUT_DIR = Path(OUTPUT_ROOT_DIR)

# 모델 파이프라인의 타입 정의 (명시적인 타입 힌트를 위해)
DiffusionPipeline = StableDiffusionPipeline


class ModelExecutor(query_pb2_grpc.QueryServicer):
    #def __init__(self):
    #    print("[인포] ModelExecutor 초기화 중...")
        # 앞서 정의한 load_diffusion_pipeline 로직 활용
    #    self.pipe, self.device = self.load_model()
    #    print(f"[서버] 준비 완료 (Device: {self.device})")
    def __init__(self):
        """
        ModelExecutor 초기화 및 모델 로드
        """
        print("[인포] ModelExecutor 초기화 및 모델 로드 중...")
        # 전역 상수 설정 반영
        self.model_directory = '/tmp/model'
        #모델명을 지정하고 테스트한다.
        self.model_file_name = "v1-5-pruned-emaonly.safetensors"
        #self.model_file_name = "sd-v1-4.safetensors" #"sd-v1-4.ckpt"
        self.ckpt_file_path = Path(self.model_directory) / self.model_file_name
        
        # 모델 로드
        self.pipe, self.device = self.load_model()
        
        if self.pipe is None:
            print("[오류] 모델 로드 실패")
        # 2. 서버 시작 시 테스트 추론 1회 실행 (Warm-up)
        #if self.pipe:
        #    self.run_startup_test()
        #else:
        #    print("[오류] 모델 로드 실패로 테스트 추론을 건너뜁니다.")

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
    """
    def load_model(self):
        print("[인포] ModelExecutor 초기화 및 모델 로드 중...")
        # 전역 상수 설정 반영
        self.model_directory = '/tmp/model'
        self.model_file_name = "v1-5-pruned-emaonly.safetensors"
        self.ckpt_file_path = Path(self.model_directory) / self.model_file_name
        
        # 모델 로드
        self.pipe, self.device = self.load_model()
        
        if self.pipe is None:
            print("[오류] 모델 로드 실패. 서버가 정상적으로 작동하지 않을 수 있습니다.")
            
        model_id = "runwayml/stable-diffusion-v1-5"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
        pipe.to(device)
        return pipe, device
    """
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
     # 1. 모델 로드 객체를 먼저 생성 (가장 중요)
    # ModelExecutor의 __init__ 내에서 load_model()이 실행되므로,
    # 여기서 model.safetensors 100% 로딩이 완료될 때까지 기다립니다.
    print("[시스템] 모델 로딩을 시작합니다. 잠시만 기다려 주세요...")
    executor = ModelExecutor()

    # 모델 로드 성공 여부 확인 (예외 처리)
    if executor.pipe is None:
        print("[치명적 오류] 모델 로드에 실패하여 서버를 시작할 수 없습니다.")
        return

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    query_pb2_grpc.add_QueryServicer_to_server(executor, server)

    port = "50051"   #"8080"
    actual_port = server.add_insecure_port(f'[::]:{port}')
    #actual_port = server.add_insecure_port(f'0.0.0.0:{port}')
    if actual_port == 0:
        print(f"[오류] 포트 {port} 바인딩에 실패했습니다! 이미 사용 중이거나 권한이 없습니다.")
        return  
    print(f"[서버] gRPC 서버 시작됨 (Port: {port})")
    server.start()
    server.wait_for_termination()

# 임시로 막는다.
if __name__ == "__main__":
    serve()