source ~/.bashrc

rm dist/* -rf
python2 setup.py bdist_wheel --universal
#python2 setup.py bdist_wheel upload

#rm dist/* -rf

python3 setup.py bdist_wheel
twine upload dist/*
