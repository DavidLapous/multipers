#prebuild
python setup.py build_ext -j4
cd build
pip wheel .. --no-build-isolation --no-deps
## This is just a double check; should be dealt with the MANIFEST.in file
zip -d multipers*.whl "*.cpp"  "*.tp" "*.h"
## repairs
pip install delocate --upgrade
delocate-listdeps multipers*.whl
delocate-wheel --require-archs x86_64 -w wheelhouse -v  multipers*.whl
mv wheelhouse/multipers*.whl ..
cd ..
