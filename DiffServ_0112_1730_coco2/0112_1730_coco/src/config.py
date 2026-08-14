# Author: KB 
# Date: 2026-03-06 
# Purpose: For keeping configuration settings 
# File: config.py 

SCALE_UP_THRESHOLD = 5
SCALE_DOWN_THRESHOLD = 1

UTILIZATION_SCALE_UP = 0.8
UTILIZATION_SCALE_DOWN = 0.3

MIN_WORKERS = 1
MAX_WORKERS = 5

AUTOSCALE_INTERVAL = 5
CHECK_INTERVAL = 5

# Options: "vertical", "horizontal", "hybrid" # hard coding scaling mode
SCALING_MODE = "horizontal"
# SCALING_MODE = "vertical"
#SCALING_MODE = "hybrid"

############################################################
# Service Level Agreement (SLA) Targets (milliseconds)
############################################################

SLA_TARGETS = {
    "interactive": 500,
    "balanced": 2000,
    "batch": 8000
}

MIN_WORKER_AGE = 60

##############################################################
# MODEL SWITCHING; Options: hot or cold; changing this
# switches the whole experiment from cold to hot or vice-versa
##############################################################

#MODEL_SWITCH_MODE = "cold"
MODEL_SWITCH_MODE = "hot" 
