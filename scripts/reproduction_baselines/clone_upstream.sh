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

mkdir -p "${CLIP_CACHE}"
if [ -f "${CLIP_CACHE}/ViT-B-16.pt" ] \
   && [ "$(sha256sum "${CLIP_CACHE}/ViT-B-16.pt" | cut -d' ' -f1)" = "${CLIP_SHA}" ]; then
    echo "CLIP ViT-B/16: already cached at ${CLIP_CACHE}"
else
    curl -fL -o "${CLIP_CACHE}/ViT-B-16.pt" "${CLIP_URL}"
    test "$(sha256sum "${CLIP_CACHE}/ViT-B-16.pt" | cut -d' ' -f1)" = "${CLIP_SHA}"
    echo "CLIP ViT-B/16: downloaded to ${CLIP_CACHE}"
fi
