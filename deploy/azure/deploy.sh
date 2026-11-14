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
#
# Resources (all in one resource group, so deleting the group removes everything):
#   Container Apps environment + app  - 2 vCPU / 4 GiB, scales to zero when idle
#   Storage account + file share       - keeps documents the admin uploads across restarts
#   Log Analytics workspace            - app logs (created by the environment)
set -euo pipefail

RG="${RG:-ragdemia-rg}"
LOCATION="${LOCATION:-centralindia}"
ENV_NAME="${ENV_NAME:-ragdemia-env}"
APP="${APP:-ragdemia}"
IMAGE="${IMAGE:-ghcr.io/shreenitha0408/ragdemia-demo:latest}"
SHARE="uploads"

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

say "Preparing the Azure CLI"
az extension add --name containerapp --upgrade --yes --only-show-errors >/dev/null
for ns in Microsoft.App Microsoft.OperationalInsights Microsoft.Storage; do
  az provider register --namespace "$ns" --wait --only-show-errors >/dev/null
done

# ---------------------------------------------------------------- update an existing app
if az containerapp show -n "$APP" -g "$RG" --only-show-errors >/dev/null 2>&1; then
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
[ -n "$LLM_KEY" ] && [ -n "$ADMIN_TOKEN" ] || { echo "Both values are required."; exit 1; }

# Student subscriptions only allow a fixed list of regions (an "allowed locations" policy).
# Read that list and pick a permitted region instead of failing half-way through.
ALLOWED=$(az policy assignment list --query "[].parameters.listOfAllowedLocations.value[]" -o tsv 2>/dev/null | tr -d '\r' | sort -u)
if [ -n "$ALLOWED" ] && ! grep -qx "$LOCATION" <<<"$ALLOWED"; then
  echo "Region '$LOCATION' is not allowed for this subscription. Allowed: $(echo $ALLOWED)"
  LOCATION=""
  for pref in southindia westindia centralindia southeastasia eastasia uaenorth japaneast koreacentral westeurope northeurope eastus eastus2 westus2; do
    if grep -qx "$pref" <<<"$ALLOWED"; then LOCATION="$pref"; break; fi
  done
  LOCATION="${LOCATION:-$(head -n1 <<<"$ALLOWED")}"
fi
say "Using region: $LOCATION"

# The resource group is only a container; reuse it if it exists (its own region does not
# restrict where the resources inside it are created).
if ! az group show -n "$RG" --only-show-errors >/dev/null 2>&1; then
  az group create -n "$RG" -l "$LOCATION" --only-show-errors >/dev/null
fi

say "Creating storage for admin uploads"
STORAGE=$(az storage account list -g "$RG" --query "[0].name" -o tsv)
if [ -z "$STORAGE" ]; then
  STORAGE="ragdemia$(openssl rand -hex 4)"
  az storage account create -n "$STORAGE" -g "$RG" -l "$LOCATION" --sku Standard_LRS \
    --kind StorageV2 --min-tls-version TLS1_2 --only-show-errors >/dev/null
fi
STORAGE_KEY=$(az storage account keys list -n "$STORAGE" -g "$RG" --query "[0].value" -o tsv)
az storage share-rm create --storage-account "$STORAGE" -g "$RG" -n "$SHARE" --quota 5 \
  --only-show-errors >/dev/null

say "Creating the Container Apps environment (2-4 minutes)"
az containerapp env create -n "$ENV_NAME" -g "$RG" -l "$LOCATION" --only-show-errors >/dev/null
az containerapp env storage set -n "$ENV_NAME" -g "$RG" --storage-name "$SHARE" \
  --azure-file-account-name "$STORAGE" --azure-file-account-key "$STORAGE_KEY" \
  --azure-file-share-name "$SHARE" --access-mode ReadWrite --only-show-errors >/dev/null
ENV_ID=$(az containerapp env show -n "$ENV_NAME" -g "$RG" --query id -o tsv)

# YAML double-quoted strings: escape backslashes and quotes in the secret values.
yaml_escape() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }
LLM_KEY_Y=$(yaml_escape "$LLM_KEY")
ADMIN_TOKEN_Y=$(yaml_escape "$ADMIN_TOKEN")

say "Creating the app (pulling a ~3 GB image, 3-6 minutes)"
SPEC=$(mktemp)
trap 'rm -f "$SPEC"' EXIT
cat > "$SPEC" <<YAML
location: $LOCATION
name: $APP
type: Microsoft.App/containerApps
properties:
  managedEnvironmentId: $ENV_ID
  configuration:
    activeRevisionsMode: Single
    ingress:
      external: true
      targetPort: 8000
      transport: auto
      allowInsecure: false
    secrets:
      - name: llm-key
        value: "$LLM_KEY_Y"
      - name: admin-token
        value: "$ADMIN_TOKEN_Y"
  template:
    containers:
      - name: ragdemia
        image: $IMAGE
        resources:
          cpu: 2.0
          memory: 4Gi
        env:
          - name: LLM_API_KEY
            secretRef: llm-key
          - name: RAGDEMIA_ADMIN_TOKEN
            secretRef: admin-token
        volumeMounts:
          - volumeName: uploads
            mountPath: /app/data/raw/uploads
        probes:
          - type: Startup
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 15
            timeoutSeconds: 5
            failureThreshold: 10
          - type: Liveness
            httpGet:
              path: /health
              port: 8000
            periodSeconds: 60
            timeoutSeconds: 5
            failureThreshold: 3
    scale:
      minReplicas: 0
      maxReplicas: 1
      rules:
        - name: http
          http:
            metadata:
              concurrentRequests: "20"
    volumes:
      - name: uploads
        storageType: AzureFile
        storageName: $SHARE
YAML
az containerapp create -n "$APP" -g "$RG" --yaml "$SPEC" --only-show-errors >/dev/null
rm -f "$SPEC"

URL=$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)
say "Done! RAGDemia is live at:  https://$URL"
echo "    The first visit starts the app (about 1 minute); after that answers take a few seconds."
echo "    It scales to zero when idle, so the first visit after a quiet period is slower."
echo "    Logs:    az containerapp logs show -n $APP -g $RG --follow"
echo "    Update:  bash deploy.sh        Remove everything:  az group delete -n $RG"
