#!/usr/bin/env bash
# Deploy (or update) the public RAGDemia demo on Azure Container Apps.
# Run it in Azure Cloud Shell (Bash):  https://shell.azure.com
#
#   curl -fsSL https://raw.githubusercontent.com/Shreenitha0408/ragdemia/main/deploy/azure/deploy.sh -o deploy.sh
#   bash deploy.sh
#
# First run: asks for your Groq API key and an admin password (typed here, stored only as
# Azure secrets), creates the resources and prints the app's HTTPS address.
# Later runs: pull the newest image (built by GitHub Actions) and roll out a new revision.
# Replace the LLM key:  bash deploy.sh --set-key
#
# Resources (all in one resource group, so deleting the group removes everything):
#   Container Apps environment + app  - 2 vCPU / 4 GiB, scales to zero when idle
#   Storage account (blob container)   - keeps documents the admin uploads across restarts
#   Log Analytics workspace            - app logs (created by the environment)
set -euo pipefail

RG="${RG:-ragdemia-rg}"
LOCATION="${LOCATION:-centralindia}"
ENV_NAME="${ENV_NAME:-ragdemia-env}"
APP="${APP:-ragdemia}"
IMAGE="${IMAGE:-ghcr.io/shreenitha0408/ragdemia-demo:latest}"

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

say "Preparing the Azure CLI"
az extension add --name containerapp --upgrade --yes --only-show-errors >/dev/null
for ns in Microsoft.App Microsoft.OperationalInsights Microsoft.Storage; do
  az provider register --namespace "$ns" --wait --only-show-errors >/dev/null
done

# ---------------------------------------------------------------- update an existing app
if az containerapp show -n "$APP" -g "$RG" --only-show-errors >/dev/null 2>&1; then
  if [ "${1:-}" = "--set-key" ]; then
    read -rsp "New Groq API key (starts with gsk_): " NEW_KEY; echo
    NEW_KEY=$(printf '%s' "$NEW_KEY" | tr -d '[:space:]')
    [ -n "$NEW_KEY" ] || { echo "No key entered."; exit 1; }
    az containerapp secret set -n "$APP" -g "$RG" --secrets "llm-key=$NEW_KEY" --only-show-errors >/dev/null
    say "API key updated"
  fi
  say "Updating $APP to the latest image"
  az containerapp update -n "$APP" -g "$RG" --image "$IMAGE" \
    --set-env-vars "DEPLOY_STAMP=$(date +%s)" --only-show-errors >/dev/null
  URL=$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)
  say "Updated. Live at: https://$URL"
  exit 0
fi

# ---------------------------------------------------------------- first deployment
say "First-time setup (your input is hidden and stored only as Azure secrets)"
read -rsp "Groq API key (starts with gsk_): " LLM_KEY; echo
read -rsp "Choose an admin password for the shared library: " ADMIN_TOKEN; echo
# Pasted keys can carry an invisible carriage return or spaces; remove them.
LLM_KEY=$(printf '%s' "$LLM_KEY" | tr -d '[:space:]')
ADMIN_TOKEN=$(printf '%s' "$ADMIN_TOKEN" | tr -d '\r\n')
[ -n "$LLM_KEY" ] && [ -n "$ADMIN_TOKEN" ] || { echo "Both values are required."; exit 1; }

# Student subscriptions only allow a fixed list of regions (an "allowed locations" policy that
# is often assigned above the subscription, where it cannot be listed). So we *probe*: try to
# create the storage account in each candidate region, nearest to India first, and keep the
# first region Azure accepts. A disallowed attempt creates nothing.
if ! az group show -n "$RG" --only-show-errors >/dev/null 2>&1; then
  az group create -n "$RG" -l "$LOCATION" --only-show-errors >/dev/null 2>&1 \
    || az group create -n "$RG" -l eastus --only-show-errors >/dev/null
fi

say "Finding a region your subscription allows, and creating storage for admin uploads"
STORAGE=$(az storage account list -g "$RG" --query "[0].name" -o tsv)
if [ -n "$STORAGE" ]; then
  LOCATION=$(az storage account show -n "$STORAGE" -g "$RG" --query location -o tsv)
else
  STORAGE="ragdemia$(openssl rand -hex 4)"
  FOUND=""
  for region in "$LOCATION" southindia centralindia westindia southeastasia eastasia uaenorth \
                koreacentral japaneast australiaeast northeurope westeurope swedencentral \
                francecentral uksouth eastus eastus2 centralus westus2 westus3 canadacentral; do
    printf '   trying %-15s ' "$region"
    if ERR=$(az storage account create -n "$STORAGE" -g "$RG" -l "$region" --sku Standard_LRS \
          --kind StorageV2 --min-tls-version TLS1_2 --only-show-errors 2>&1 >/dev/null); then
      echo "ok"; FOUND="$region"; break
    elif grep -q "RequestDisallowedByAzure" <<<"$ERR"; then
      echo "not allowed"
    else
      echo "failed"; echo "$ERR"; exit 1
    fi
  done
  [ -n "$FOUND" ] || { echo "No allowed region found. Check Azure portal > Policy > Assignments."; exit 1; }
  LOCATION="$FOUND"
fi
say "Using region: $LOCATION"
# Admin uploads are kept in a Blob Storage container (works with every Container Apps
# environment type, unlike file-share mounts).
STORAGE_CONN=$(az storage account show-connection-string -n "$STORAGE" -g "$RG" --query connectionString -o tsv)
az storage container create --name uploads --connection-string "$STORAGE_CONN" \
  --only-show-errors >/dev/null

say "Creating the Container Apps environment (2-4 minutes)"
if ! az containerapp env show -n "$ENV_NAME" -g "$RG" --only-show-errors >/dev/null 2>&1; then
  az containerapp env create -n "$ENV_NAME" -g "$RG" -l "$LOCATION" --only-show-errors >/dev/null
fi

say "Creating the app (pulling a ~3 GB image, 3-6 minutes)"
create_app() {  # $1 = vCPU, $2 = memory
  az containerapp create -n "$APP" -g "$RG" --environment "$ENV_NAME" --image "$IMAGE" \
    --ingress external --target-port 8000 --cpu "$1" --memory "$2" \
    --min-replicas 0 --max-replicas 1 \
    --secrets "llm-key=$LLM_KEY" "admin-token=$ADMIN_TOKEN" "storage-conn=$STORAGE_CONN" \
    --env-vars "LLM_API_KEY=secretref:llm-key" "RAGDEMIA_ADMIN_TOKEN=secretref:admin-token" \
               "AZURE_STORAGE_CONNECTION_STRING=secretref:storage-conn" \
    --only-show-errors >/dev/null
}
if ! create_app 2.0 4Gi; then
  echo "   2 vCPU / 4 GiB was not accepted here; retrying with 1 vCPU / 2 GiB"
  create_app 1.0 2Gi
fi

URL=$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)
say "Done! RAGDemia is live at:  https://$URL"
echo "    The first visit starts the app (about 1 minute); after that answers take a few seconds."
echo "    It scales to zero when idle, so the first visit after a quiet period is slower."
echo "    Logs:    az containerapp logs show -n $APP -g $RG --follow"
echo "    Update:  bash deploy.sh        Remove everything:  az group delete -n $RG"
