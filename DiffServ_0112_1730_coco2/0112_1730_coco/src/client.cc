#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <chrono> // 시간 측정 라이브러리 추가
#include <iomanip> // 소수점 제어를 위해 필요

#include <grpcpp/grpcpp.h>

#include "query.grpc.pb.h"
#include "infaas_request_status.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using infaas::internal::Query;
using infaas::internal::QueryOnlineRequest;
using infaas::internal::QueryOnlineResponse;

class QueryClient {
public:
    QueryClient(std::shared_ptr<Channel> channel)
        : stub_(Query::NewStub(channel)) {}

    void RequestImageGeneration() {
        QueryOnlineRequest request;
        QueryOnlineResponse response;
        ClientContext context;

        // 1. 프롬프트 리스트 준비 (client.py 스타일)
        std::vector<std::string> prompts = {
            "A rainbow on the moutain", 
            "A beautlful woman in the sunset"
        };

        // 2. 요청 데이터 설정
        // C++ gRPC에서 repeated 필드는 add_필드명()을 사용해 하나씩 추가합니다. 
        for (const auto& p : prompts) {
            request.add_prompt(p); 
        }

        request.set_steps(25);                                     // [cite: 6]
        request.set_cfg_scale(7.5f);                               // [cite: 6]
        request.set_batchsize(static_cast<int32_t>(prompts.size())); // 리스트 크기에 맞춤 
        request.set_seed(42);                                      // 

        std::cout << "[클라이언트] 요청 전송: " << prompts.size() << " 개의 프롬프트" << std::endl;
        
        // --- 시간 측정 시작 ---
        auto start = std::chrono::high_resolution_clock::now();

        // 3. QueryOnline RPC 호출 
        Status status = stub_->QueryOnline(&context, request, &response);

        // --- 시간 측정 종료 ---
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        // 소수점 4자리까지 출력하도록 설정
        std::cout << "------------------------------------------------------------" << std::endl;
        std::cout << "[성능 로그] 총 추론 시간: " << std::fixed << std::setprecision(4)
                << elapsed.count() << " 초 (CPU+GPU)" << std::endl;
        std::cout << "------------------------------------------------------------" << std::endl;


        // 4. 응답 결과 처리 [cite: 13, 14]
        if (status.ok()) {
            if (response.status().status() == 1) { // 1: SUCCESS [cite: 2]
                std::cout << "✅ 서버 응답 수신 성공" << std::endl;
                
                int path_count = response.image_paths_size();
                std::cout << "- 생성된 이미지 개수: " << path_count << "개" << std::endl;

                for (int i = 0; i < path_count; ++i) {
                    std::cout << "  [" << i + 1 << "] 저장 경로: " << response.image_paths(i) << std::endl;
                }
            } else {
                std::cout << "⚠️ 서버 처리 실패: " << response.status().msg() << std::endl;
            }
        } else {
            std::cout << "❌ gRPC 오류 발생: " << status.error_code() << " - " << status.error_message() << std::endl;
        }
    }

private:
    std::unique_ptr<Query::Stub> stub_;
};

int main(int argc, char** argv) {
    std::string target_str = "localhost:50051";
    QueryClient client(grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));
    
    client.RequestImageGeneration();
    return 0;
}