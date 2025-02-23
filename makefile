.ONESHELL: 

SHELL = /bin/zsh

stage: 
	python scripts/prep-qmd.py
	quarto render --profile publish
	python scripts/create-ipynb.py
	python scripts/insert-colab-link.py

publish: 
	python scripts/prep-qmd.py
	quarto render --profile publish
	python scripts/create-ipynb.py
	python scripts/insert-colab-link.py
	git add .
	git commit -m "Update"
	git push

prep: 
	python scripts/create-ipynb.py
	python scripts/prep-qmd.py

preview: 
	quarto preview --profile preview

clean: 
	find . -type f -name "* [0-9]*" -delete
	find . -name "* [0-9]*" -type d -exec rm -r "{}" \;
	rm -rf docs	
	rm -rf chapters

