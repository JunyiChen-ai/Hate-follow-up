#!/usr/bin/env bash
# Clone the two baseline repositories, pinned, into third_party/.
#
# third_party/ is gitignored and stays pristine: nothing in this study edits
# it. The modified copies live under scripts/reproduction_baselines/ and every
# difference is listed in PATCHES.md. Re-running this script is safe; it skips
# a clone that is already at the pinned commit.
#
# Also fetches the frozen CLIP ViT-B/16 checkpoint that both models load for
# their text encoder. The visual features this study consumes were extracted
# with the HuggingFace mirror of the same weights (openai/clip-vit-base-patch16,
# post-projection image_embeds), so the visual and text embeddings live in one
# space.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THIRD_PARTY="${REPO_ROOT}/third_party"
CLIP_CACHE="${CLIP_CACHE:-${HOME}/.cache/clip}"

VADCLIP_URL="https://github.com/nwpu-zxr/VadCLIP.git"
VADCLIP_SHA="c41067f07d252efcda18008bea367886070c33b0"
DSANET_URL="https://github.com/lessiYin/DSANet.git"
DSANET_SHA="eb335b23fd6f01810bcd176c948c10348764a504"
CMHKF_URL="https://github.com/ssp-seven/CMHKF.git"
CMHKF_SHA="3b07707f240892ef1284dcbad5fac96fc8504c70"
FED_WSVAD_URL="https://github.com/wbfwonderful/Fed-WSVAD.git"
FED_WSVAD_SHA="287747f5d7cb0d52e3f0667885de78bb9a61b139"
VERA_URL="https://github.com/vera-framework/VERA.git"
VERA_SHA="15b8bcb8574a977c229c577f50bfe6f06d07106e"
VADR1_URL="https://github.com/wbfwonderful/Vad-R1.git"
VADR1_SHA="8536296b748d389dfca2d8f81a9703aa57404bc2"
EVENTVAD_URL="https://github.com/YihuaJerry/EventVAD.git"
EVENTVAD_SHA="25cacd88a82af389776d2b397239f39961ac2d27"
LAVAD_URL="https://github.com/lucazanella/lavad.git"
LAVAD_SHA="1ad46c666d1b3cfb262f3dd84769acf873285056"
# EventVAD's two dependencies that ship as source rather than as packages.
# RAFT is pinned to the head of princeton-vl/RAFT because the repository has
# had one functional commit since 2021 and no tags; the checkpoint, not the
# code, is what the port is sensitive to and its sha256 is checked below.
RAFT_URL="https://github.com/princeton-vl/RAFT.git"
RAFT_SHA="2888e15a51fa41140771d3f498ed8023cff098d1"
VIDEOLLAMA2_URL="https://github.com/DAMO-NLP-SG/VideoLLaMA2.git"
# The commit EventVAD's src/score/requirements.txt pins:
#   -e git+https://github.com/DAMO-NLP-SG/VideoLLaMA2.git@c0bb03ab...
VIDEOLLAMA2_SHA="c0bb03abf6b8a6b9a8dccac006fb4db5d4d9e414"

CLIP_URL="https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
CLIP_SHA="5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"

clone_pinned () {
    local name="$1" url="$2" sha="$3" dest="${THIRD_PARTY}/$1"
    if [ -d "${dest}/.git" ] && [ "$(git -C "${dest}" rev-parse HEAD)" = "${sha}" ]; then
        echo "${name}: already at ${sha:0:7}"
        return
    fi
    rm -rf "${dest}"
    git clone "${url}" "${dest}"
    git -C "${dest}" checkout --quiet "${sha}"
    echo "${name}: cloned at ${sha:0:7}"
}

mkdir -p "${THIRD_PARTY}"
clone_pinned VadCLIP "${VADCLIP_URL}" "${VADCLIP_SHA}"
clone_pinned DSANet  "${DSANET_URL}"  "${DSANET_SHA}"
clone_pinned CMHKF "${CMHKF_URL}" "${CMHKF_SHA}"
clone_pinned Fed-WSVAD "${FED_WSVAD_URL}" "${FED_WSVAD_SHA}"
clone_pinned VERA "${VERA_URL}" "${VERA_SHA}"
# Vad-R1 is read, not run: vadr1/run_vadr1_inference.py carries its own copy of
# the released prompt and its --verify-prompt flag compares that copy against
# this clone. No CLIP checkpoint is involved; the model is the released
# Qwen2.5-VL-7B fine-tune on HuggingFace, fetched separately with
#     hf download wbfwonderful/Vad-R1 --local-dir /home/jehc223/data/checkpoints/vad_r1
clone_pinned Vad-R1  "${VADR1_URL}"  "${VADR1_SHA}"
# EventVAD is read, not run -- see DESIGN_EVENTVAD.md for why it cannot be:
# the release imports a `graph_propagation` it never defines. Its RAFT and
# VideoLLaMA2 dependencies are run, so both are cloned.
clone_pinned EventVAD    "${EVENTVAD_URL}"    "${EVENTVAD_SHA}"
clone_pinned lavad       "${LAVAD_URL}"       "${LAVAD_SHA}"
clone_pinned RAFT        "${RAFT_URL}"        "${RAFT_SHA}"
clone_pinned VideoLLaMA2 "${VIDEOLLAMA2_URL}" "${VIDEOLLAMA2_SHA}"

mkdir -p "${CLIP_CACHE}"
if [ -f "${CLIP_CACHE}/ViT-B-16.pt" ] \
   && [ "$(sha256sum "${CLIP_CACHE}/ViT-B-16.pt" | cut -d' ' -f1)" = "${CLIP_SHA}" ]; then
    echo "CLIP ViT-B/16: already cached at ${CLIP_CACHE}"
else
    curl -fL -o "${CLIP_CACHE}/ViT-B-16.pt" "${CLIP_URL}"
    test "$(sha256sum "${CLIP_CACHE}/ViT-B-16.pt" | cut -d' ' -f1)" = "${CLIP_SHA}"
    echo "CLIP ViT-B/16: downloaded to ${CLIP_CACHE}"
fi

# ------------------------------------------------------------------ EventVAD
# RAFT ships its checkpoints as one Dropbox zip, fetched by the repository's
# own download_models.sh. `raft-things` is the entry EventVAD's placeholder
# path names. The zip carries five checkpoints and 82 MB total, so it is
# unpacked whole and the one that matters is checksummed.
RAFT_DIR="${RAFT_DIR:-/home/jehc223/data/checkpoints/raft}"
RAFT_MODELS_URL="https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip"
RAFT_THINGS_SHA="fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1"

mkdir -p "${RAFT_DIR}"
if [ -f "${RAFT_DIR}/raft-things.pth" ] \
   && [ "$(sha256sum "${RAFT_DIR}/raft-things.pth" | cut -d' ' -f1)" = "${RAFT_THINGS_SHA}" ]; then
    echo "RAFT things: already at ${RAFT_DIR}"
else
    curl -fL -o "${RAFT_DIR}/models.zip" "${RAFT_MODELS_URL}"
    unzip -o -j "${RAFT_DIR}/models.zip" -d "${RAFT_DIR}"
    rm -f "${RAFT_DIR}/models.zip"
    test "$(sha256sum "${RAFT_DIR}/raft-things.pth" | cut -d' ' -f1)" = "${RAFT_THINGS_SHA}"
    echo "RAFT things: downloaded to ${RAFT_DIR}"
fi

# VideoLLaMA2.1-7B-16F, about 16 GB. The SigLIP tower ships inside it, but
# `SiglipImageProcessor.from_pretrained` and `SiglipVisionConfig.from_pretrained`
# still resolve the tower by hub name, so the two small config files are
# pulled into the HF cache to keep the scoring stage offline-safe.
VL2_DIR="${VL2_DIR:-/home/jehc223/data/checkpoints/videollama2}"
if [ -f "${VL2_DIR}/model.safetensors.index.json" ]; then
    echo "VideoLLaMA2.1-7B-16F: already at ${VL2_DIR}"
else
    hf download DAMO-NLP-SG/VideoLLaMA2.1-7B-16F --local-dir "${VL2_DIR}"
    echo "VideoLLaMA2.1-7B-16F: downloaded to ${VL2_DIR}"
fi
hf download google/siglip-so400m-patch14-384 --include "*.json" > /dev/null
echo "SigLIP so400m config: cached"
