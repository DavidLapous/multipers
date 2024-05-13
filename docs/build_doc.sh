rm -rf source 
sphinx-apidoc -f -o source .. ../setup.py
rm -rf _build 
make html
