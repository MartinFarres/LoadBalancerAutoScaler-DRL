#!/bin/bash

echo "[1/3] Cazando procesos huerfanos de Python..."
pkill -9 -f uvicorn
pkill -9 -f locust
pkill -9 -f tensorboard
pkill -9 -f train_agent.py
pkill -9 -f test_agent.py
pkill -9 -f main.py

echo "[2/3] Liberando puertos..."
fuser -k -9 8000/tcp 2>/dev/null
fuser -k -9 6006/tcp 2>/dev/null
fuser -k -9 8089/tcp 2>/dev/null

echo "[3/3] Purgando contenedores Docker zombies..."
docker rm -f $(docker ps -a -q --filter "name=lbas") 2>/dev/null

docker network rm lbas_network 2>/dev/null

echo "Terminado"