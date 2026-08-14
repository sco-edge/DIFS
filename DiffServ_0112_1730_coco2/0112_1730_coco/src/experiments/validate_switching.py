import os
import sys
import csv
import time
from datetime import datetime

import grpc

import query_pb2
import query_pb2_grpc

# ------------------------------------------------------------------
# Allowing the file to import from outside its folder
# ------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

SRC_DIR = os.path.abspath(
    os.path.join(CURRENT_DIR, "..")
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config
import experiment_models

MODEL_ROOT = os.environ.get(
    "DIFS_MODEL_DIR",
    "/tmp/model"
)


#########################################################################
# Timestamp
#########################################################################

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_DIR = os.path.join(RESULT_DIR, timestamp)

os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_FILE = os.path.join(

    OUTPUT_DIR,

    "switch_validation.csv"

)

#########################################################################
# CSV Header
#########################################################################

with open(CSV_FILE, "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow([
        "ModelID",
        "DisplayName",
        "SwitchMode",
        "StartTime",
        "FinishTime",
        "SwitchTime",
        "RPCSuccess",
        "WorkerModel",
        "Verified"
    ])

#########################################################################
# Worker
#########################################################################

worker = f"{WORKER_HOST}:{WORKER_PORT}"
print()
print("="*70)
print("Connecting to")
print(worker)
print("="*70)
channel = grpc.insecure_channel(worker)
stub = query_pb2_grpc.QueryStub(channel)

#########################################################################
# Helper
#########################################################################

def switch_model(model):

    start = time.perf_counter()
    response = stub.LoadModel(
        query_pb2.LoadModelRequest(
            model_id=model.model_id
        )
    )

    finish = time.perf_counter()
    return response, finish-start

#########################################################################
# Validation
#########################################################################

for model in MODELS:

    print()
    print("="*70)
    print("Switching to")
    print(model.display_name)
    print("="*70)
    start_wall = time.time()
    response, switch_time = switch_model(model)
    finish_wall = time.time()
    status = stub.WorkerStatus(
        query_pb2.WorkerStatusRequest()
    )

    worker_model = status.model_name
    verified = worker_model == model.model_name
    print()
    print("RPC Success :", response.success)
    print("Worker Model:", worker_model)
    print("Expected    :", model.model_name)
    print("Verified    :", verified)
    print("Switch Time :", switch_time)

    with open(CSV_FILE,"a",newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            model.model_id,
            model.display_name,
            config.MODEL_SWITCH_MODE,
            start_wall,
            finish_wall,
            switch_time,
            response.success,
            worker_model,
            verified
        ])

    time.sleep(SWITCH_DELAY)

#########################################################################

print()
print("="*70)
print("Validation Finished")
print(CSV_FILE)
print("="*70)