#!/usr/bin/env bash
# Historical baseline source bootstrap into third_party/.
#
# third_party/ is gitignored and stays pristine: nothing in this study edits
# it. The modified copies live under scripts/reproduction_baselines/ and every
# difference is listed in PATCHES.md. Re-running this script is safe; it skips
# an existing source checkout. Git identifiers are reserved for multi-machine code sync.
#
# Also fetches the frozen CLIP ViT-B/16 checkpoint that both models load for
# their text encoder. The visual features this study consumes were extracted
# with the HuggingFace mirror of the same weights (openai/clip-vit-base-patch16,
# post-projection image_embeds), so the visual and text embeddings live in one
# space.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THIRD_PARTY="${REPO_ROOT}/third_party"
CLIP_CACHE="${CLIP_CACHE:-${REPO_ROOT}/data/assets/clip}"

VADCLIP_URL="https://github.com/nwpu-zxr/VadCLIP.git"
DSANET_URL="https://github.com/lessiYin/DSANet.git"
CMHKF_URL="https://github.com/ssp-seven/CMHKF.git"
FED_WSVAD_URL="https://github.com/wbfwonderful/Fed-WSVAD.git"
VERA_URL="https://github.com/vera-framework/VERA.git"
VADR1_URL="https://github.com/wbfwonderful/Vad-R1.git"
EVENTVAD_URL="https://github.com/YihuaJerry/EventVAD.git"
LAVAD_URL="https://github.com/lucazanella/lavad.git"
# EventVAD's two dependencies that ship as source rather than as packages.
# RAFT supplies EventVAD optical-flow weights; availability is validated by
# actual model loading in the consuming extractor, without content digests.
RAFT_URL="https://github.com/princeton-vl/RAFT.git"
VIDEOLLAMA2_URL="https://github.com/DAMO-NLP-SG/VideoLLaMA2.git"

CLIP_URL="https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"

clone_source () {
    local name="$1" url="$2" dest="${THIRD_PARTY}/$1"
    if [ -d "${dest}/.git" ]; then
        echo "${name}: existing source at ${dest}"
        return
    fi
    if [ -e "${dest}" ]; then
        echo "${name}: destination exists without a git checkout: ${dest}" >&2
        return 1
    fi
    git clone "${url}" "${dest}"
    echo "${name}: cloned source to ${dest}"
}

mkdir -p "${THIRD_PARTY}"
clone_source VadCLIP "${VADCLIP_URL}"
clone_source DSANet "${DSANET_URL}"
clone_source CMHKF "${CMHKF_URL}"
clone_source Fed-WSVAD "${FED_WSVAD_URL}"
clone_source VERA "${VERA_URL}"
clone_source Vad-R1 "${VADR1_URL}"
clone_source EventVAD "${EVENTVAD_URL}"
clone_source lavad "${LAVAD_URL}"
clone_source RAFT "${RAFT_URL}"
clone_source VideoLLaMA2 "${VIDEOLLAMA2_URL}"

mkdir -p "${CLIP_CACHE}"
if [ -s "${CLIP_CACHE}/ViT-B-16.pt" ]; then
    echo "CLIP ViT-B/16: already cached at ${CLIP_CACHE}"
else
    curl -fL -o "${CLIP_CACHE}/ViT-B-16.pt" "${CLIP_URL}"
    echo "CLIP ViT-B/16: downloaded to ${CLIP_CACHE}"
fi

# ------------------------------------------------------------------ EventVAD
# RAFT ships its checkpoints as one Dropbox zip, fetched by the repository's
# own download_models.sh. `raft-things` is the entry EventVAD's placeholder
# path names. The zip carries five checkpoints and 82 MB total, so it is
# unpacked whole; the consuming extractor parses the needed checkpoint.
RAFT_DIR="${RAFT_DIR:-${REPO_ROOT}/data/assets/raft}"
RAFT_MODELS_URL="https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip"

mkdir -p "${RAFT_DIR}"
if [ -s "${RAFT_DIR}/raft-things.pth" ]; then
    echo "RAFT things: already at ${RAFT_DIR}"
else
    curl -fL -o "${RAFT_DIR}/models.zip" "${RAFT_MODELS_URL}"
    unzip -o -j "${RAFT_DIR}/models.zip" -d "${RAFT_DIR}"
    rm -f "${RAFT_DIR}/models.zip"
    echo "RAFT things: downloaded to ${RAFT_DIR}"
fi

# VideoLLaMA2.1-7B-16F, about 16 GB. The SigLIP tower ships inside it, but
# `SiglipImageProcessor.from_pretrained` and `SiglipVisionConfig.from_pretrained`
# still resolve the tower by hub name, so the two small config files are
# pulled into the HF cache to keep the scoring stage offline-safe.
VL2_DIR="${VL2_DIR:-${REPO_ROOT}/data/assets/videollama2}"
if [ -f "${VL2_DIR}/model.safetensors.index.json" ]; then
    echo "VideoLLaMA2.1-7B-16F: already at ${VL2_DIR}"
else
    hf download DAMO-NLP-SG/VideoLLaMA2.1-7B-16F --local-dir "${VL2_DIR}"
    echo "VideoLLaMA2.1-7B-16F: downloaded to ${VL2_DIR}"
fi
hf download google/siglip-so400m-patch14-384 --include "*.json" > /dev/null
echo "SigLIP so400m config: cached"
