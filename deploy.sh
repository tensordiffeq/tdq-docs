cd tensordiffeq
rm -r /_build/
jupyter-book build tensordiffeq/
git add .
git commit -m "commit from bash"
git push
cd tensordiffeq/
ghp-import -n -p -f _build/html