#!/bin/bash

# Author: KB(GPT)
# Purpose: To start DIFS

echo "Stopping INFaaS..."

pkill -f modelreg_server || true
pkill -f queryfe_server || true
pkill -f query_executor || true
pkill redis-server || true

sleep 2
echo "INFaaS stopped."
