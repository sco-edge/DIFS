import asyncio
import grpc
import grpc.aio  # 비동기 gRPC 라이브러리
import query_pb2
import query_pb2_grpc
import infaas_request_status_pb2
import time
import os
import json
import argparse
import random

# --- 유틸리티 함수 ---
def load_json(file_path):
    """JSON 파일을 로드합니다."""
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_address_by_id(data, target_id):
    """ID에 해당하는 address를 반환합니다."""
    for item in data:
        if item.get("id") == target_id:
            return item.get("address")
    return None

async def run():
    # 1. 실행 인자 파싱 (Q1, Q2 입력을 받기 위함)
    parser = argparse.ArgumentParser(description="gRPC Async Client")
    parser.add_argument("-query", type=str, required=True, help="접속 타겟 (예: Q1, Q2)")
    parser.add_argument("-steps", type=int, default=25, help="추론 단계 수 (Timestep)")
    parser.add_argument("-sampler", type=int, default=1, help="사용할 샘플러 ID")
    args = parser.parse_args()

    print(f"[steps: {args.steps}  [sampler ID: {args.sampler}")

    # 2. Q1, Q2 문자열에서 숫자 ID 추출
    try:
        # "Q1" -> 1, "Q2" -> 2로 변환
        target_id = int(args.query.upper().replace("Q", ""))
    except ValueError:
        print(f"[오류] 잘못된 입력입니다: {args.query}. Q1 또는 Q2 형식을 사용하세요.")
        return

    # 3. address_port.json 로드 및 주소 결정
    address_data = load_json("address_port.json")
    if not address_data:
        print("[오류] address_port.json 파일을 찾을 수 없거나 형식이 잘못되었습니다.")
        return

    server_address = get_address_by_id(address_data, target_id)
    if not server_address:
        print(f"[오류] ID {target_id}에 해당하는 서버 주소가 json에 없습니다.")
        return

    print(f"[클라이언트] {args.query} 타겟 접속 시도: {server_address}")

    # 4. 서버 채널 연결 (대용량 이미지 수신을 위해 메시지 크기 제한 확장 권장)
    options = [
        ('grpc.max_receive_message_length', 100 * 1024 * 1024) # 100MB까지 허용
    ]
    ##with grpc.insecure_channel(server_address) as channel:  #'localhost:50052'
    # grpc.aio.insecure_channel을 사용하여 비동기 채널을 생성합니다.
    async with grpc.aio.insecure_channel(server_address, options=options) as channel:    
        stub = query_pb2_grpc.QueryStub(channel)
        
        #prompts = ["A rainbow on the moutain", "A beautlful woman in the sunset"]
        #prompts = ["dog", "dog", "dog", "dog", "dog", "car", "car", "car", "car", "car"]
        #prompts = ["a rainbow","a rainbow","a rainbow","a rainbow","a rainbow", "a rainbow","a rainbow","a rainbow","a rainbow","a rainbow", "a rainbow","a rainbow","a rainbow","a rainbow","a rainbow" ]
        prompts = ["dog"] * 10000
        #prompts = ["dog", "cat"] 
        #prompts = extract_coco_prompts('annotations/captions_val2014.json', 10)

        # 2. QueryOnlineRequest 생성
        # repeated 필드는 파이썬의 list를 그대로 대입하면 됩니다.
        request = query_pb2.QueryOnlineImageRequest(
            Prompt=prompts,           # ["sentence1", "sentence2"] 형식
            Steps=args.steps,         # 입력받은 timestep 값 적용 #25
            Sampler_Type=args.sampler,     # 입력받은 sampler id 적용 (proto 파일에 정의되어 있어야 함)
            CFG_Scale=7.5,
            BatchSize=len(prompts),   # 프롬프트 개수에 맞춰 BatchSize 설정
            Seed=42
            #model=["stable-diffusion-v1-5"],
            #submitter="user_test"
        )
        print(f"[클라이언트] 비동기 스트리밍 요청 전송... '{request.Prompt}'")
        
        try:
            start_time = time.perf_counter()
            image_count = 0
            save_dir = "./client_received"
            os.makedirs(save_dir, exist_ok=True)

            # [변경] 비동기 스트리밍 호출 (서버가 yield로 보내는 응답 수신 준비)
            # stream은 비동기 이터레이터가 됩니다.
            stream = stub.QueryOnlineImage(request)

            # [변경] async for를 사용하여 이미지가 도착할 때마다 즉시 처리합니다.
            async for response in stream:
                if response.status.status == 1:  # SUCCESS
                    image_count += 1
                    
                    # 수신된 이미지 데이터 처리 (리스트의 첫 번째 요소 사용)
                    if response.image_data:
                        img_bytes = response.image_data   #[0]
                        file_name = f"received_{args.query}_{image_count}_{int(time.time())}.png"
                        save_path = os.path.join(save_dir, file_name)
                        
                        # 파일 저장
                        with open(save_path, "wb") as f:
                            f.write(img_bytes)
                        
                        elapsed = time.perf_counter() - start_time
                        print(f"  [{image_count}] 이미지 수신 및 저장 완료 ({elapsed:.2f}s): {save_path}")
                else:
                    print(f"[클라이언트] 서버 처리 실패: {response.status.msg}")
                    break

            print(f"\n[완료] 총 {image_count}개의 이미지를 비동기로 수신했습니다.")
            print(f"[성능 로그] 최종 소요 시간: {time.perf_counter() - start_time:.4f} 초")
            
        except grpc.RpcError as e:
            # RPC 오류 발생 시에도 시간 출력 (타임아웃 확인용)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"[클라이언트] gRPC 비동기 오류 발생: {e.code()} - {e.details()}")
            print(f"[성능 로그] 오류 발생 시점까지의 시간: {elapsed_time:.4f} 초")

if __name__ == "__main__":
    # [변경] asyncio.run()을 통해 비동기 루프를 시작합니다.
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n클라이언트를 종료합니다.")
    ##run()
