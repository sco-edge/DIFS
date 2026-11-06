# Author: KB 
# Date: 2026-07-20
# Purpose: To trigger horizontal scaling; lunch many client requests concurrently. Autscaler
# can then respond by lunching multiple workers to handle the requests.
# File: stress_test.py


import asyncio
import argparse
import grpc

import query_pb2
import query_pb2_grpc


async def send_request(
    worker_id,
    host,
    port,
    prompt,
    steps,
    sampler,
):

    async with grpc.aio.insecure_channel(
        f"{host}:{port}",
        options=[
            ("grpc.max_receive_message_length", 100 * 1024 * 1024)
        ],
    ) as channel:

        stub = query_pb2_grpc.QueryStub(channel)

        request = query_pb2.QueryOnlineImageRequest(
            Prompt=[prompt],
            Steps=steps,
            Sampler_Type=sampler,
            CFG_Scale=7.5,
            BatchSize=1,
            Seed=42,
        )

        try:

            async for _ in stub.QueryOnlineImage(request):
                pass

            print(f"[CLIENT {worker_id}] finished")

        except Exception as e:

            print(f"[CLIENT {worker_id}] ERROR: {e}")


async def main(args):

    tasks = []

    for i in range(args.requests):

        tasks.append(

            asyncio.create_task(

                send_request(
                    i,
                    args.host,
                    args.port,
                    f"Prompt {i}",
                    args.steps,
                    args.sampler,
                )

            )

        )

        #
        # arrival interval
        #
        await asyncio.sleep(args.interval)

    await asyncio.gather(*tasks)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50050)

    parser.add_argument("--requests", type=int, default=100)

    parser.add_argument("--interval", type=float, default=0.10)

    parser.add_argument("--steps", type=int, default=20)

    parser.add_argument("--sampler", type=int, default=1)

    args = parser.parse_args()

    asyncio.run(main(args))
