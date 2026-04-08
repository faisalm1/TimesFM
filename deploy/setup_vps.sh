#!/bin/bash
# Run this once on the VPS as root to install + start the app.
set -e

APP_DIR="/opt/timesfm"
REPO="https://github.com/faisalm1/TimesFM.git"
SERVICE_USER="root"

echo "=== [1/6] Installing system packages ==="
apt-get update -qq
apt-get install -y git python3.11 python3.11-venv python3-pip python3.11-dev \
    build-essential libgomp1 nginx curl

echo "=== [2/6] Cloning repo ==="
if [ -d "$APP_DIR" ]; then
    cd "$APP_DIR" && git pull
else
    git clone "$REPO" "$APP_DIR"
fi
cd "$APP_DIR"

echo "=== [3/6] Creating virtualenv + installing deps ==="
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas pyarrow python-dotenv alpaca-py pytz \
    fastapi "uvicorn[standard]" lightgbm scikit-learn joblib apscheduler
pip install timesfm torch --index-url https://download.pytorch.org/whl/cpu
pip install -e . --no-build-isolation

echo "=== [4/6] Creating data directories ==="
mkdir -p data/cache data/ml_artifacts

echo "=== [5/6] Installing systemd service ==="
cat > /etc/systemd/system/timesfm.service << 'EOF'
[Unit]
Description=TimesFM Gap Dashboard API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/timesfm
EnvironmentFile=/opt/timesfm/.env
ExecStart=/opt/timesfm/.venv/bin/python -m uvicorn gap_dashboard.api:app --app-dir src --host 0.0.0.0 --port 8001
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable timesfm
systemctl restart timesfm

echo "=== [6/6] Done! ==="
echo "API running at http://$(curl -s ifconfig.me):8001"
echo "Logs: journalctl -u timesfm -f"
