#!/usr/bin/env bash
# Install or update the public RAGDemia demo on an Ubuntu server (Oracle Cloud Always Free).
#
#   First time:  bash install.sh          (asks for your Groq API key and an admin password)
#   Update:      bash install.sh          (pulls the latest code and rebuilds; keeps your data)
#
# What it does: installs Docker, opens ports 80/443 in the server's own firewall, writes the
# secrets to deploy/oracle/.env (readable only by you), and starts the app behind HTTPS.
set -euo pipefail

REPO_URL="https://github.com/Shreenitha0408/ragdemia.git"
APP_DIR="$HOME/ragdemia"

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

say "Installing Docker and git"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo -E apt-get install -y -qq docker.io docker-compose-v2 git curl iptables-persistent >/dev/null
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER" || true

say "Opening ports 80 and 443 in the server firewall"
# Oracle's Ubuntu images block everything except SSH with iptables; allow web traffic.
for port in 80 443; do
  sudo iptables -C INPUT -p tcp --dport "$port" -m state --state NEW -j ACCEPT 2>/dev/null \
    || sudo iptables -I INPUT 6 -p tcp --dport "$port" -m state --state NEW -j ACCEPT
done
sudo netfilter-persistent save >/dev/null

say "Fetching the code"
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" pull --ff-only
else
  git clone "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR/deploy/oracle"

if [ ! -f .env ]; then
  say "First-time configuration (input is not shown and not stored anywhere else)"
  read -rsp "Groq API key (starts with gsk_): " LLM_API_KEY; echo
  read -rsp "Choose an admin password (used to manage the shared library): " ADMIN_TOKEN; echo
  IP=$(curl -fsS https://ifconfig.me || curl -fsS https://api.ipify.org)
  DEFAULT_DOMAIN="${IP//./-}.sslip.io"
  read -rp "Domain [$DEFAULT_DOMAIN]: " DOMAIN
  DOMAIN=${DOMAIN:-$DEFAULT_DOMAIN}
  umask 077
  cat > .env <<CONF
DOMAIN=$DOMAIN
LLM_API_KEY=$LLM_API_KEY
ADMIN_TOKEN=$ADMIN_TOKEN
CONF
fi
mkdir -p "$APP_DIR/data"

say "Building and starting (first build takes 10-15 minutes)"
sudo docker compose --env-file .env up -d --build

DOMAIN=$(grep '^DOMAIN=' .env | cut -d= -f2)
say "Done. RAGDemia will be live at:  https://$DOMAIN"
echo "    The first start downloads the sample papers and builds the index (~3-5 min)."
echo "    Follow progress with:  cd $APP_DIR/deploy/oracle && sudo docker compose logs -f app"
