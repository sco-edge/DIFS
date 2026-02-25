import grpc
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

def extract_coco_prompts(json_path, count=10000):
    with open(json_path, 'r') as f:
        data = json.load(f)
   
    # 모든 캡션 문장만 추출
    all_captions = [ann['caption'] for ann in data['annotations']]
   
    # 중복을 피하기 위해 무작위로 count만큼 선택
    selected_prompts = random.sample(all_captions, count)
   
    return selected_prompts

def run():
    # 1. 실행 인자 파싱 (Q1, Q2 입력을 받기 위함)
    parser = argparse.ArgumentParser(description="gRPC Client with Dynamic Address")
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
    with grpc.insecure_channel(server_address) as channel:  #'localhost:50052'    
        stub = query_pb2_grpc.QueryStub(channel)
        
        #prompts = ["A rainbow on the moutain", "A beautlful woman in the sunset"]
        #prompts = ["dog", "dog", "dog", "dog", "dog", "car", "car", "car", "car", "car"]
        #prompts = ["dog","dog"]
        prompts = extract_coco_prompts('annotations/captions_val2014.json', 10)

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
        print(f"[클라이언트] 요청 전송 (QueryOnlineImage): '{request.Prompt}'")
        
        try:
            # --- 시간 측정 시작 ---
            start_time = time.perf_counter()

            # 3. RPC 호출 및 응답 수신 [cite: 5, 13, 14]
            response = stub.QueryOnlineImage(request)

            # --- 시간 측정 종료 및 계산 ---
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            # 4. 상태 확인 및 결과 출력 
            if response.status.status == 1: # SUCCESS = 1 [cite: 2]
                print(f"[클라이언트] 서버 응답 수신 성공")
                print(f"[성능 로그] 총 쿼리 소요 시간: {elapsed_time:.4f} 초 (RTT)")
                #print(f"- 생성된 이미지 개수: {len(response.image_paths)}개")
                
                # --- [추가 포인트] 수신된 바이너리 데이터를 파일로 저장 ---
                save_dir = "./client_received"
                os.makedirs(save_dir, exist_ok=True)

                print(f"- 수신된 이미지 개수: {len(response.image_data)}개")
               
                for i, img_bytes in enumerate(response.image_data):
                    # 파일명 결정 (서버 경로에서 파일명만 추출하거나 새로 명명)
                    file_name = f"received_{args.query}_{i+1}_{int(time.time())}.png"
                    save_path = os.path.join(save_dir, file_name)
                   
                    # 바이너리 쓰기 모드('wb')로 파일 저장
                    with open(save_path, "wb") as f:
                        f.write(img_bytes)
                    print(f"  [{i+1}] 파일 저장 완료: {save_path}")
            else:
                print(f"[클라이언트] 서버 처리 실패: {response.status.msg}")
            
        except grpc.RpcError as e:
            # RPC 오류 발생 시에도 시간 출력 (타임아웃 확인용)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"[클라이언트] gRPC 오류 발생: {e.code()} - {e.details()}")
            print(f"[성능 로그] 오류 발생 시점까지의 시간: {elapsed_time:.4f} 초")

if __name__ == "__main__":
    run()