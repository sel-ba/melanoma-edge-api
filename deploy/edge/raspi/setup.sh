#!/bin/bash
set -euo pipefail

echo "=== Melanoma Detection API — Raspberry Pi 4 Setup ==="

ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "ERROR: This script is for ARM64 (aarch64). Detected: $ARCH"
    exit 1
fi

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
fi

echo "Pulling edge image..."
docker pull ghcr.io/yourusername/melanoma-api:latest-arm64

# Create systemd service for auto-start
sudo tee /etc/systemd/system/melanoma-api.service > /dev/null << 'EOF'
[Unit]
Description=Melanoma Detection API
After=docker.service
Requires=docker.service

[Service]
Restart=always
RestartSec=10
ExecStart=/usr/bin/docker run --rm \
    --name melanoma-api \
    -p 8080:8080 \
    --memory="1g" \
    --cpus="2" \
    ghcr.io/yourusername/melanoma-api:latest-arm64

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable melanoma-api
sudo systemctl start melanoma-api

echo "Service started. Check: systemctl status melanoma-api"
echo "API available at: http://$(hostname -I | awk '{print $1}'):8080"
