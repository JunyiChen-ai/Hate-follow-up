#!/usr/bin/env bash
# Build the camera-ready sources without latexmk's content-digest database.
set -euo pipefail
source_dir=$(realpath "$1")
mkdir -p "$2"
build_dir=$(realpath "$2")
exec >"$build_dir/run.log" 2>&1
hostname
echo "$$" >"$build_dir/run.pid"
printf 'Source: %s\nOutput: %s\n' "$source_dir" "$build_dir"
date -Is
cd "$source_dir"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" main.tex
(
    cd "$build_dir"
    BIBINPUTS="$source_dir:" BSTINPUTS="$source_dir:" bibtex main
)
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" main.tex
echo DONE
