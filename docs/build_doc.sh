rm -rf source 
sphinx-apidoc -f -P -o source ../multipers ../setup.py
rm -rf _build 
make html
