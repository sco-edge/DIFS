import grpc
import query_pb2
import query_pb2_grpc

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

SRC_DIR = os.path.abspath(
    os.path.join(CURRENT_DIR, "..")
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config
import experiment_models

channel = grpc.insecure_channel("localhost:50060")

stub = query_pb2_grpc.QueryStub(channel)

response = stub.LoadModel(

    query_pb2.LoadModelRequest(

        model_id=2

    )

)

print(response)