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
    # channel = grpc.insecure_channel('192.168.128.8:50010') # PNB(ip address of lab GPU-based machine together with port on which the server is running)
    channel = grpc.insecure_channel("127.0.0.1:50051")
    stub = query_pb2_grpc.QueryServiceStub(channel)

	 request = query_pb2.QueryOnlineRequest(
        model_name="stable-diffusion-v1",
        task="DIFFUSION",
        prompt="a futuristic city at sunset",
        steps=30,
        guidance_scale=7.5,
        width=512,
        height=512
    )

    response = stub.QueryOnline(request)

    if response.status != "OK":
        print("Error:", response.error_message)
        return

    with open("output.png", "wb") as f:
        f.write(response.image_png)

    print("Image saved to output.png")

if __name__ == "__main__":
    run()
