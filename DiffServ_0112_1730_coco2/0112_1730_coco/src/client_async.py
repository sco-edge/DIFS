import asyncio
import grpc
import grpc.aio  # asynchronous gRPC library
import query_pb2
import query_pb2_grpc
import infaas_request_status_pb2
import time
import os
import json
import argparse
import random

# --- utility functions ---
def load_json(file_path):
    """Loads a JSON file."""
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_address_by_id(data, target_id):
    """Returns the address corresponding to the ID."""
    for item in data:
        if item.get("id") == target_id:
            return item.get("address")
    return None

async def run():
    # 1. parse command-line arguments (to accept Q1, Q2 inputs)
    parser = argparse.ArgumentParser(description="gRPC Async Client")
    parser.add_argument("-query", type=str, required=True, help="connection target (e.g., Q1, Q2)")
    parser.add_argument("-steps", type=int, default=25, help="number of inference steps (Timestep)")
    parser.add_argument("-sampler", type=int, default=1, help="sampler ID to use")
    args = parser.parse_args()

    print(f"[steps: {args.steps}  [sampler ID: {args.sampler}")

    # 2. extract the numeric ID from the Q1/Q2 string
    try:
        # convert "Q1" -> 1, "Q2" -> 2
        target_id = int(args.query.upper().replace("Q", ""))
    except ValueError:
        print(f"[ERROR] Invalid input: {args.query}. Use the Q1 or Q2 format.")
        return

    # 3. load address_port.json and determine the address
    address_data = load_json("address_port.json")
    if not address_data:
        print("[ERROR] The address_port.json file could not be found or its format is invalid.")
        return

    server_address = get_address_by_id(address_data, target_id)
    if not server_address:
        print(f"[ERROR] No server address for ID {target_id} exists in the JSON.")
        return

    print(f"[CLIENT] Attempting to connect to target {args.query}: {server_address}")

    # 4. connect the server channel (consider increasing the message size limit for receiving large images)
    options = [
        ('grpc.max_receive_message_length', 100 * 1024 * 1024) # allow up to 100MB
    ]
    ##with grpc.insecure_channel(server_address) as channel:  #'localhost:50052'
    # Use grpc.aio.insecure_channel to create an asynchronous channel.
    async with grpc.aio.insecure_channel(server_address, options=options) as channel:    
        stub = query_pb2_grpc.QueryStub(channel)
        
        #prompts = ["A rainbow on the moutain", "A beautlful woman in the sunset"]
        #prompts = ["dog", "dog", "dog", "dog", "dog", "car", "car", "car", "car", "car"]
        #prompts = ["a rainbow","a rainbow","a rainbow","a rainbow","a rainbow", "a rainbow","a rainbow","a rainbow","a rainbow","a rainbow", "a rainbow","a rainbow","a rainbow","a rainbow","a rainbow" ]
        prompts = ["dog"] * 10000
        #prompts = ["dog", "cat"] 
        #prompts = extract_coco_prompts('annotations/captions_val2014.json', 10)

        # 2. create the QueryOnlineRequest
        # repeated fields accept a Python list directly.
        request = query_pb2.QueryOnlineImageRequest(
            Prompt=prompts,           # in the form ["sentence1", "sentence2"]
            Steps=args.steps,         # apply the received timestep value #25
            Sampler_Type=args.sampler,     # apply the received sampler id (must be defined in the proto file)
            CFG_Scale=7.5,
            BatchSize=len(prompts),   # set the BatchSize to match the number of prompts
            Seed=42
            #model=["stable-diffusion-v1-5"],
            #submitter="user_test"
        )
        print(f"[CLIENT] Sending asynchronous streaming request... '{request.Prompt}'")
        
        try:
            start_time = time.perf_counter()
            image_count = 0
            save_dir = "./client_received"
            os.makedirs(save_dir, exist_ok=True)

            # [CHANGED] asynchronous streaming call (ready to receive responses sent by the server via yield)
            # the stream becomes an asynchronous iterator.
            stream = stub.QueryOnlineImage(request)

            # [CHANGED] use async for to process each image immediately as it arrives.
            async for response in stream:
                if response.status.status == 1:  # SUCCESS
                    image_count += 1
                    
                    # process the received image data (use the first element of the list)
                    if response.image_data:
                        img_bytes = response.image_data   #[0]
                        file_name = f"received_{args.query}_{image_count}_{int(time.time())}.png"
                        save_path = os.path.join(save_dir, file_name)
                        
                        # save the file
                        with open(save_path, "wb") as f:
                            f.write(img_bytes)
                        
                        elapsed = time.perf_counter() - start_time
                        print(f"  [{image_count}] Image received and saved ({elapsed:.2f}s): {save_path}")
                else:
                    print(f"[CLIENT] Server processing failed: {response.status.msg}")
                    break

            print(f"\n[DONE] Asynchronously received {image_count} images in total.")
            print(f"[PERFORMANCE LOG] Total elapsed time: {time.perf_counter() - start_time:.4f} s")
            
        except grpc.RpcError as e:
            # print the time even when an RPC error occurs (to check timeouts)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"[CLIENT] An asynchronous gRPC error occurred: {e.code()} - {e.details()}")
            print(f"[PERFORMANCE LOG] Elapsed time until the error occurred: {elapsed_time:.4f} s")

if __name__ == "__main__":
    # [CHANGED] start the async loop via asyncio.run().
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nShutting down the client.")
    ##run()
