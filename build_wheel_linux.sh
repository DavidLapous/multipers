##prebuild
python setup.py build_ext -j4
cd build
pip wheel .. --no-build-isolation --no-deps
## This is just a double check; should be dealt with the MANIFEST.in file
zip -d multipers*.whl "*.cpp"  "*.tp" "*.h"
## repairs
pip install auditwheel --upgrade
auditwheel show multipers*.whl
auditwheel repair multipers*.whl --plat $(auditwheel show multipers*.whl | grep -o 'manylinux.*"' | tr -d '"')
mv wheelhouse/multipers*.whl ..
cd ..
