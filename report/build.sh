#!/usr/bin/env bash
# Build report.pdf. Requires pdflatex (TeX Live). Runs twice to resolve refs.
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
echo "Built: $(pwd)/report.pdf"
