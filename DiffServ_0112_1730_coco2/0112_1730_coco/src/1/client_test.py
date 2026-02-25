import grpc
import query_pb2
import query_pb2_grpc
import infaas_request_status_pb2
import time

def run():
    # 1. 서버 채널 연결
    with grpc.insecure_channel('192.168.128.8:50051') as channel:  #8080
        stub = query_pb2_grpc.QueryStub(channel)
        
        prompts = ["A rainbow on the moutain", "A beautlful woman in the sunset"]

        # 2. QueryOnlineRequest 생성
        # repeated 필드는 파이썬의 list를 그대로 대입하면 됩니다.
        request = query_pb2.QueryOnlineRequest(
            Prompt=prompts,           # ["sentence1", "sentence2"] 형식
            Steps=25,
            CFG_Scale=7.5,
            BatchSize=len(prompts),   # 프롬프트 개수에 맞춰 BatchSize 설정
            Seed=42
            #model=["stable-diffusion-v1-5"],
            #submitter="user_test"
        )
        print(f"[클라이언트] 요청 전송: '{request.Prompt}'")
        
        try:
            # --- 시간 측정 시작 ---
            start_time = time.perf_counter()

            # 3. RPC 호출 및 응답 수신 [cite: 5, 13, 14]
            response = stub.QueryOnline(request)

            # --- 시간 측정 종료 및 계산 ---
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            # 4. 상태 확인 및 결과 출력 
            if response.status.status == 1: # SUCCESS = 1 [cite: 2]
                print(f"[클라이언트] 서버 응답 수신 성공")
                print(f"[성능 로그] 총 쿼리 소요 시간: {elapsed_time:.4f} 초 (RTT)")
                print(f"- 생성된 이미지 개수: {len(response.image_paths)}개")
                
                # 경로 배열(repeated string) 출력
                for i, path in enumerate(response.image_paths):
                    print(f"  [{i+1}] 저장 경로: {path}")
            else:
                print(f"[클라이언트] 서버 처리 실패: {response.status.msg}")
                print(f"[성능 로그] 실패까지 걸린 시간: {elapsed_time:.4f} 초")
            
        except grpc.RpcError as e:
            # RPC 오류 발생 시에도 시간 출력 (타임아웃 확인용)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"[클라이언트] gRPC 오류 발생: {e.code()} - {e.details()}")
            print(f"[성능 로그] 오류 발생 시점까지의 시간: {elapsed_time:.4f} 초")

if __name__ == "__main__":
    run()
