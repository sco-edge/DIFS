#!/bin/bash
echo "Stopping local INFaaS..."
pkill -f modelreg_server || true
pkill -f queryfe_server || true
pkill -f worker_daemon || true
pkill redis-server || true
echo "All stopped."
