import grpc
import query_pb2
import query_pb2_grpc
import infaas_request_status_pb2

def check_server_health():
    # 1. Connecting to the Server Channel (Default Port 50060; originally8080)
    with grpc.insecure_channel('localhost:50060') as channel:
        stub = query_pb2_grpc.QueryStub(channel)
        
        # 2. create the HeartbeatRequest object
        # HeartbeatRequest has a status field.
        request = query_pb2.HeartbeatRequest()
        
        print("[Client] Checking server status (Heartbeat)")
        
        try:
            # 3. Heartbeat RPC Call 
            response = stub.Heartbeat(request)
            
            # 4. Analysis of Server Response Results
            status_value = response.status.status
            
            if status_value == infaas_request_status_pb2.SUCCESS:
                print("✅ Server Status: Normal (Model loaded)")
            elif status_value == infaas_request_status_pb2.UNAVAILABLE:
                print("⚠️ Server Status: Preparing (Model not loaded or loading)")
            else:
                print(f"❓ Server Status: Unknown (Status Code: {status_value})")
                
        except grpc.RpcError as e:
            # gRPC If the connection itself fails (e.g., the server is down)
            print(f"❌ Failed to connect to the server: {e.code()} - {e.details()}")

if __name__ == "__main__":
    check_server_health()