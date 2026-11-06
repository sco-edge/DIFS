#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <chrono> // added time measurement library
#include <iomanip> // needed for decimal-place control

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

        // 1. prepare the prompt list (client.py style)
        std::vector<std::string> prompts = {
            "A rainbow on the moutain", 
            "A beautlful woman in the sunset"
        };

        // 2. set the request data
        // in C++ gRPC, use add_fieldname() to add repeated fields one by one. 
        for (const auto& p : prompts) {
            request.add_prompt(p); 
        }

        request.set_steps(25);                                     // [cite: 6]
        request.set_cfg_scale(7.5f);                               // [cite: 6]
        request.set_batchsize(static_cast<int32_t>(prompts.size())); // match the list size 
        request.set_seed(42);                                      // 

        std::cout << "[CLIENT] sending request: " << prompts.size() << "  prompts" << std::endl;
        
        // --- start timing ---
        auto start = std::chrono::high_resolution_clock::now();

        // 3. make the QueryOnline RPC call 
        Status status = stub_->QueryOnline(&context, request, &response);

        // --- stop timing ---
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        // configure output to show 4 decimal places
        std::cout << "------------------------------------------------------------" << std::endl;
        std::cout << "[PERFORMANCE LOG] total inference time: " << std::fixed << std::setprecision(4)
                << elapsed.count() << " s (CPU+GPU)" << std::endl;
        std::cout << "------------------------------------------------------------" << std::endl;


        // 4. process the response [cite: 13, 14]
        if (status.ok()) {
            if (response.status().status() == 1) { // 1: SUCCESS [cite: 2]
                std::cout << "✅ Server response received successfully" << std::endl;
                
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