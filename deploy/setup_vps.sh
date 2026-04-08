#!/bin/bash
# Run this once on the VPS as root to install + start the app.
set -e

APP_DIR="/opt/timesfm"
REPO="https://github.com/faisalm1/TimesFM.git"

echo "=== [1/7] Installing system packages ==="
apt-get update -qq
apt-get install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update -qq
apt-get install -y python3.11 python3.11-venv python3.11-dev \
    build-essential libgomp1 nginx curl git

echo "=== [2/7] Checking repo ==="
if [ -d "$APP_DIR/.git" ]; then
    cd "$APP_DIR" && git pull
else
    git clone "$REPO" "$APP_DIR"
fi
cd "$APP_DIR"

echo "=== [3/7] Creating virtualenv + installing deps ==="
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas pyarrow python-dotenv alpaca-py pytz \
    fastapi "uvicorn[standard]" lightgbm scikit-learn joblib apscheduler
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install timesfm
pip install -e . --no-build-isolation

echo "=== [4/7] Creating data directories ==="
mkdir -p data/cache data/ml_artifacts

echo "=== [5/7] Creating .env (if missing) ==="
if [ ! -f .env ]; then
    cp .env.example .env
    echo ">>> Created .env from example — edit /opt/timesfm/.env with real keys!"
fi

echo "=== [6/7] Installing systemd service ==="
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

echo "=== [7/7] Setting up nginx reverse proxy ==="
cat > /etc/nginx/sites-available/timesfm << 'NGINX'
server {
    listen 80;
    server_name _;

    location /api/ {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }

    location / {
        root /opt/timesfm/web/dist;
        try_files $uri $uri/ /index.html;
    }
}
NGINX

ln -sf /etc/nginx/sites-available/timesfm /etc/nginx/sites-enabled/timesfm
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl restart nginx

echo ""
echo "=== DONE ==="
VPS_IP=$(curl -s ifconfig.me)
echo "Dashboard: http://$VPS_IP"
echo "API:       http://$VPS_IP/api/health"
echo "Logs:      journalctl -u timesfm -f"
echo ""
echo "NEXT: Edit /opt/timesfm/.env with your real API keys!"
