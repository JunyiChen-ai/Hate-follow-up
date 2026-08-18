"""Vendored upstream pieces shared by both baseline ports.

clip/     verbatim from VadCLIP @ c41067f src/clip (byte-identical in DSANet)
tools.py  verbatim from VadCLIP @ c41067f src/utils/tools.py
layers.py from VadCLIP @ c41067f src/utils/layers.py, two patches (see PATCHES.md)
data.py   new: corpus manifests, labels, dataset, validation carve
runtime.py new: shared training/inference plumbing
"""
