#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>

// 컴파일된 proto 헤더 파일들을 포함합니다.
#include "query.grpc.pb.h"
#include "infaas_request_status.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using infaas::internal::Query;
using infaas::internal::HeartbeatRequest;
using infaas::internal::HeartbeatResponse;

class QueryClient {
public:
    QueryClient(std::shared_ptr<Channel> channel)
        : stub_(Query::NewStub(channel)) {}

    // 서버 상태 확인 함수
    void CheckServerHealth() {
        HeartbeatRequest request;
        HeartbeatResponse response;
        ClientContext context;

        std::cout << "[클라이언트] 서버 상태(Heartbeat) 확인 중..." << std::endl;

        // 3. Heartbeat RPC 호출 [cite: 5]
        Status status = stub_->Heartbeat(&context, request, &response);

        if (status.ok()) {
            // 4. 서버 응답 결과 분석
            // proto 정의에 따라 response.status()를 통해 InfaasRequestStatus에 접근합니다. 
            int status_value = response.status().status();

            if (status_value == 1) { // 1: SUCCESS 
                std::cout << "✅ 서버 상태: 정상 (모델 로드 완료)" << std::endl;
            } else if (status_value == 3) { // 3: UNAVAILABLE (InfaasRequestStatus 정의 기준) 
                std::cout << "⚠️ 서버 상태: 준비 중 (모델 미로드 또는 로딩 중)" << std::endl;
            } else {
                std::cout << "❓ 서버 상태: 알 수 없음 (Status Code: " << status_value << ")" << std::endl;
            }
        } else {
            // gRPC 통신 자체가 실패한 경우
            std::cout << "❌ 서버 연결 실패: " << status.error_code() 
                      << " - " << status.error_message() << std::endl;
        }
    }

private:
    std::unique_ptr<Query::Stub> stub_;
};

int main(int argc, char** argv) {
    // 1. 서버 채널 연결 (50051 포트)
    std::string target_str = "localhost:50051";
    QueryClient client(grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));
    
    client.CheckServerHealth();

    return 0;
}