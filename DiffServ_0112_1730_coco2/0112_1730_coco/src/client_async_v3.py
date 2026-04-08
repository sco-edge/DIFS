# Date: 2026.03.18
# Purpose:
# Adapted: from 'client_async_v2.py'

import asyncio
import grpc
import grpc.aio
import query_pb2
import query_pb2_grpc
import infaas_request_status_pb2
import time
import os
import json
import argparse
import random


# --- 유틸리티 함수 ---
def extract_coco_prompts(json_path, count=10):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_annotations = data.get('annotations', [])

    if len(all_annotations) >= count:
        sampled_annos = random.sample(all_annotations, count)
    else:
        sampled_annos = all_annotations

    return [anno['caption'] for anno in sampled_annos]


############################################################
# 🔥 MAIN ASYNC CLIENT
############################################################

async def run():

    parser = argparse.ArgumentParser(description="gRPC Async Client (Autoscaler Mode)")

    # 🔥 KEEP same interface (Q1 etc.), but no longer used for routing
    parser.add_argument("-query", type=str, required=True)

    parser.add_argument("-steps", type=int, default=25)
    parser.add_argument("-sampler", type=int, default=1)

    # 🔥 NEW: scheduler endpoint
    parser.add_argument("-host", type=str, default="localhost")
    parser.add_argument("-port", type=int, default=50050)

    args = parser.parse_args()

    print(f"[CLIENT] Steps: {args.steps}, Sampler: {args.sampler}")
    print(f"[CLIENT] Connecting to scheduler at {args.host}:{args.port}")

    ############################################################
    # CONNECT TO SCHEDULER (NOT WORKERS)
    ############################################################

    options = [
        ('grpc.max_receive_message_length', 100 * 1024 * 1024)
    ]

    async with grpc.aio.insecure_channel(f"{args.host}:{args.port}", options=options) as channel:

        # ⚠️ IMPORTANT: must match server.py service name
        stub = query_pb2_grpc.QueryStub(channel)

        ############################################################
        # LOAD PROMPTS
        ############################################################

        json_file = 'annotations/captions_val2014.json'
        # prompt_count = 10000
        prompt_count = 5

        try:
            prompts = extract_coco_prompts(json_file, prompt_count)

            print(f"✅ Loaded {len(prompts)} prompts")

        except FileNotFoundError:
            print(f"❌ File not found: {json_file}")
            return

        ############################################################
        # BUILD REQUEST
        ############################################################

        request = query_pb2.QueryOnlineImageRequest(
            Prompt=prompts,
            Steps=args.steps,
            Sampler_Type=args.sampler,
            CFG_Scale=7.5,
            #BatchSize=len(prompts),
            BatchSize=1,
            Seed=42
        )

        print(f"[CLIENT] Sending request to scheduler...")

        ############################################################
        # STREAMING RESPONSE
        ############################################################

        try:
            start_time = time.perf_counter()
            image_count = 0

            save_dir = "./client_received"
            os.makedirs(save_dir, exist_ok=True)

            stream = stub.QueryOnlineImage(request)

            async for response in stream:

                if response.status.status == 1:

                    image_count += 1

                    if response.image_data:

                        img_bytes = response.image_data

                        file_name = f"received_{image_count}_{int(time.time())}.png"
                        save_path = os.path.join(save_dir, file_name)

                        with open(save_path, "wb") as f:
                            f.write(img_bytes)

                        elapsed = time.perf_counter() - start_time

                        print(f"[{image_count}] Saved ({elapsed:.2f}s): {save_path}")

                else:
                    print(f"[CLIENT] Server error: {response.status.msg}")
                    break

            ############################################################
            # SUMMARY
            ############################################################

            total_time = time.perf_counter() - start_time

            print("\n[CLIENT] ===== SUMMARY =====")
            print(f"Images received: {image_count}")
            print(f"Total time: {total_time:.2f}s")
            if image_count > 0:
                print(f"Avg latency/image: {total_time / image_count:.2f}s")

        except grpc.RpcError as e:

            elapsed_time = time.perf_counter() - start_time

            print(f"[CLIENT] gRPC Error: {e.code()} - {e.details()}")
            print(f"[CLIENT] Time before failure: {elapsed_time:.2f}s")


############################################################
# ENTRY
############################################################

if __name__ == "__main__":

    try:
        asyncio.run(run())

    except KeyboardInterrupt:
        print("\n[CLIENT] Interrupted and exiting.")
