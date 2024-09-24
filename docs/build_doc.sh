rm -rf source 
sphinx-apidoc -f -P -o source ../multipers
rm -rf _build 
make html
