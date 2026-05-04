#!/usr/bin/env sh
set -eu

rm -rf source
sphinx-apidoc -f -P -o source ../multipers
rm -rf _build
sphinx-build -M html . _build ${SPHINXOPTS:-} ${O:-}
