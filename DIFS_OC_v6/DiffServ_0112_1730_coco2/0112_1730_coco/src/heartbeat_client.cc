#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>

// include the compiled proto header files.
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

    // function to check the server status
    void CheckServerHealth() {
        HeartbeatRequest request;
        HeartbeatResponse response;
        ClientContext context;

        std::cout << "[CLIENT] checking the server status (Heartbeat)..." << std::endl;

        // 3. make the Heartbeat RPC call [cite: 5]
        Status status = stub_->Heartbeat(&context, request, &response);

        if (status.ok()) {
            // 4. analyze the server response
            // access InfaasRequestStatus via response.status() according to the proto definition. 
            int status_value = response.status().status();

            if (status_value == 1) { // 1: SUCCESS 
                std::cout << "✅ server status: normal (model loaded)" << std::endl;
            } else if (status_value == 3) { // 3: UNAVAILABLE (per the InfaasRequestStatus definition) 
                std::cout << "⚠️ server status: preparing (model not loaded or still loading)" << std::endl;
            } else {
                std::cout << "❓ 서버 상태: 알 수 없음 (Status Code: " << status_value << ")" << std::endl;
            }
        } else {
            // when the gRPC communication itself fails
            std::cout << "❌ server connection failed: " << status.error_code() 
                      << " - " << status.error_message() << std::endl;
        }
    }

private:
    std::unique_ptr<Query::Stub> stub_;
};

int main(int argc, char** argv) {
    // 1. connect the server channel (port 50051)
    std::string target_str = "localhost:50051";
    QueryClient client(grpc::CreateChannel(target_str, grpc::InsecureChannelCredentials()));
    
    client.CheckServerHealth();

    return 0;
}