#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install GLM-5.1 contrib model into the NXDI venv.
# Copies dev source to the venv site-packages so the model can be imported
# directly from neuronx_distributed_inference.models.glm5.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference}"
SITE_PACKAGES="$VENV/lib/python3.*/site-packages/neuronx_distributed_inference/models"

# Find the actual site-packages path (glob expansion)
TARGET_DIR=$(echo $SITE_PACKAGES)
if [ ! -d "$TARGET_DIR" ]; then
    echo "ERROR: Could not find NXDI models directory at $SITE_PACKAGES"
    echo "Make sure the venv is installed at $VENV"
    exit 1
fi

DEST="$TARGET_DIR/glm5"
echo "Installing GLM-5.1 to $DEST ..."

# Create target directory
mkdir -p "$DEST"

# Copy source files
cp "$SCRIPT_DIR/src/modeling_glm5.py" "$DEST/"
cp "$SCRIPT_DIR/src/rope_util.py" "$DEST/"
cp "$SCRIPT_DIR/src/__init__.py" "$DEST/"

echo "GLM-5.1 installed successfully."
echo ""
echo "Usage:"
echo "  from neuronx_distributed_inference.models.glm5 import NeuronGlm5ForCausalLM, Glm5InferenceConfig"
