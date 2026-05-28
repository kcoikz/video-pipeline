#!/bin/bash
# Launch a video-pipeline pod — tries all regions until one succeeds.
# Usage: RUNPOD_KEY=xxx ./launch_pod.sh
# Outputs: POD_ID and POD_URL on success

RUNPOD_KEY="${RUNPOD_KEY:?Need to set RUNPOD_KEY env var}"
IMAGE="ghcr.io/kcoikz/video-pipeline-pub:latest"
REGISTRY_AUTH="cmpoqe1h500cajr0756c7qngj"

# Pairs: "REGION:VOLUME_ID" — avoids declare -A (macOS bash 3.2 compat)
REGION_VOLUMES=(
  "EU-RO-1:tifjm0naz9"
  "EU-CZ-1:idsera2r8h"
  "US-TX-3:s87itjzpfj"
)

GPU_TYPES=(
  "NVIDIA GeForce RTX 4090"
  "NVIDIA GeForce RTX 3090"
  "NVIDIA L40S"
  "NVIDIA L4"
  "NVIDIA RTX A6000"
  "NVIDIA RTX A5000"
)

launch() {
  local REGION=$1
  local VOL_ID=$2
  local GPU=$3

  RESULT=$(curl -s -X POST "https://api.runpod.io/graphql" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $RUNPOD_KEY" \
    -d "{
      \"query\": \"mutation { podFindAndDeployOnDemand(input: {
        cloudType: SECURE,
        gpuCount: 1,
        volumeInGb: 0,
        containerDiskInGb: 40,
        minVcpuCount: 4,
        minMemoryInGb: 16,
        gpuTypeId: \\\"$GPU\\\",
        name: \\\"video-pipeline\\\",
        imageName: \\\"$IMAGE\\\",
        dockerArgs: \\\"\\\",
        ports: \\\"5002/http,8004/http,8188/http\\\",
        volumeMountPath: \\\"/workspace\\\",
        networkVolumeId: \\\"$VOL_ID\\\",
        containerRegistryAuthId: \\\"$REGISTRY_AUTH\\\",
        env: [{ key: \\\"HF_HOME\\\", value: \\\"/workspace/.hf_cache\\\" }]
      }) { id } }\"
    }" 2>/dev/null)

  echo "$RESULT" | python3 -c "
import sys, json
d = json.load(sys.stdin)
pod = d.get('data', {}).get('podFindAndDeployOnDemand')
print(pod['id'] if pod else 'FAIL')
" 2>/dev/null
}

echo "Trying to launch pod across all regions..."

for RV in "${REGION_VOLUMES[@]}"; do
  REGION="${RV%%:*}"
  VOL_ID="${RV##*:}"
  for GPU in "${GPU_TYPES[@]}"; do
    echo "  [$REGION] $GPU ..."
    POD_ID=$(launch "$REGION" "$VOL_ID" "$GPU")
    if [ "$POD_ID" != "FAIL" ] && [ "$POD_ID" != "None" ] && [ -n "$POD_ID" ]; then
      POD_URL="https://${POD_ID}-5002.proxy.runpod.net"
      echo ""
      echo "SUCCESS"
      echo "  Region:  $REGION"
      echo "  GPU:     $GPU"
      echo "  Pod ID:  $POD_ID"
      echo "  URL:     $POD_URL"
      echo ""
      echo "POD_ID=$POD_ID"
      echo "POD_URL=$POD_URL"
      exit 0
    fi
  done
done

echo "FAILED: no GPU available in any region"
exit 1
