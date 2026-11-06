# Author: Mr. Heesik
# Purpose: To test that DIFS client can communicate with server

# PNB (2025.01.06)
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),"../.."))
PROTO_PATH = os.path.join(PROJECT_ROOT, "protos","python_protos")

sys.path.insert(0,PROTO_PATH)


import grpc
import query_pb2
import query_pb2_grpc
import infaas_request_status_pb2
import time
import os

def run():
    # 1. 서버 채널 연결 (대용량 이미지 수신을 위해 메시지 크기 제한 확장 권장)
    options = [
        ('grpc.max_receive_message_length', 100 * 1024 * 1024) # allow up to 100MB
    ]
    #with grpc.insecure_channel('localhost:50051') as channel:
    with grpc.insecure_channel('192.168.128.8:50010') as channel:
        stub = query_pb2_grpc.QueryStub(channel)

        #prompts = ["A rainbow on the moutain", "A beautlful woman in the sunset"]
        prompts = ["dog", "dog", "dog", "dog", "dog", "car", "car", "car", "car", "car"]
        #prompts = ["take a rest"]

        # 2. create the QueryOnlineRequest
        # repeated fields accept a Python list directly.
#        request = query_pb2.QueryOnlineImageRequest(
        request = query_pb2.QueryOnlineRequest(
            Prompt=prompts,           # in the form ["sentence1", "sentence2"]
            Steps=25,
            CFG_Scale=7.5,
            BatchSize=len(prompts),   # set the BatchSize to match the number of prompts
            Seed=42
            #model=["stable-diffusion-v1-5"],
            #submitter="user_test"
        )
        print(f"[CLIENT] Sending request (QueryOnlineImage): '{request.Prompt}'")

        try:
            # --- start timing ---
            start_time = time.perf_counter()

            # 3. make the RPC call and receive the response [cite: 5, 13, 14]
            # response = stub.QueryOnlineImage(request)
            response = stub.QueryOnline(request)

            # --- stop timing and compute ---
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            # 4. check the status and print the result
            if response.status.status == 1: # SUCCESS = 1 [cite: 2]
                print(f"[CLIENT] Server response received successfully")
                print(f"[PERFORMANCE LOG] Total query time: {elapsed_time:.4f} s (RTT)")
                #print(f"- number of generated images: {len(response.image_paths)}")

                # --- [EXTRA POINT] save the received binary data to a file ---
                save_dir = "./client_received"
                os.makedirs(save_dir, exist_ok=True)



				# print(f"- number of received images: {len(response.image_data)}")

                # for i, img_bytes in enumerate(response.image_data):
                #     # determine the filename (either extract it from the server path or assign a new name)
                #     file_name = f"received_image_{i+1}_{int(time.time())}.png"
                #     save_path = os.path.join(save_dir, file_name)

                    # # save the file in binary write mode ('wb')
                    # with open(save_path, "wb") as f:
                        # f.write(img_bytes)
                        
                    # print(f"  [{i+1}] File saved: {save_path}")                        
                
                print(f"- number of received images: {len(response.image_paths)}")

                for i, img_path in enumerate(response.image_paths):
                    print(f"  [{i+1}] Server image path: {img_path}")        
                        
                        
                        


            else:
                print(f"[CLIENT] Server processing failed: {response.status.msg}")

        except grpc.RpcError as e:
            # print the time even when an RPC error occurs (to check timeouts)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"[CLIENT] A gRPC error occurred: {e.code()} - {e.details()}")
            print(f"[PERFORMANCE LOG] Elapsed time until the error occurred: {elapsed_time:.4f} s")

if __name__ == "__main__":
    run()
