rm -rf source 
sphinx-apidoc -f -P -o source .. ../setup.py
rm -rf _build 
make html
