# Author: KB 
# Date: 2026-03-06 
# Purpose: Start autoscaler during system initialization 
# File: server.py 




from autoscaler import AutoScaler

autoscaler = AutoScaler(
    scheduler,
    worker_pool,
    config
)

autoscaler.start()
