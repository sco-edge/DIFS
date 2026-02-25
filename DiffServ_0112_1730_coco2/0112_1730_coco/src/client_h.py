import grpc
import query_pb2
import query_pb2_grpc
import infaas_request_status_pb2

def check_server_health():
    # 1. 서버 채널 연결 (기본 8080 포트)
    with grpc.insecure_channel('localhost:8080') as channel:
        stub = query_pb2_grpc.QueryStub(channel)
        
        # 2. HeartbeatRequest 객체 생성
        # HeartbeatRequest는 status 필드를 가집니다.
        request = query_pb2.HeartbeatRequest()
        
        print("[클라이언트] 서버 상태(Heartbeat) 확인 중...")
        
        try:
            # 3. Heartbeat RPC 호출 
            response = stub.Heartbeat(request)
            
            # 4. 서버 응답 결과 분석
            status_value = response.status.status
            
            if status_value == infaas_request_status_pb2.SUCCESS:
                print("✅ 서버 상태: 정상 (모델 로드 완료)")
            elif status_value == infaas_request_status_pb2.UNAVAILABLE:
                print("⚠️ 서버 상태: 준비 중 (모델 미로드 또는 로딩 중)")
            else:
                print(f"❓ 서버 상태: 알 수 없음 (Status Code: {status_value})")
                
        except grpc.RpcError as e:
            # gRPC 통신 자체가 실패한 경우 (서버가 꺼져 있는 등)
            print(f"❌ 서버 연결 실패: {e.code()} - {e.details()}")

if __name__ == "__main__":
    check_server_health()