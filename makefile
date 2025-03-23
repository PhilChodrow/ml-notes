.ONESHELL: 

SHELL = /bin/zsh

PYTHON = env/bin/python3

stage: 
	$(PYTHON) scripts/prep-qmd.py
	quarto render --profile publish
	$(PYTHON) scripts/create-ipynb.py
	$(PYTHON) scripts/insert-colab-link.py

publish: 
	$(PYTHON) scripts/prep-qmd.py
	quarto render --profile publish
	$(PYTHON) scripts/create-ipynb.py
	$(PYTHON) scripts/insert-colab-link.py
	git add .
	git commit -m "Update"
	git push

prep: 
	$(PYTHON) scripts/create-ipynb.py
	$(PYTHON) scripts/prep-qmd.py

preview: 
	quarto preview --profile preview

clean: 
	find . -type f -name "* [0-9]*" -delete
	find . -name "* [0-9]*" -type d -exec rm -r "{}" \;
	rm -rf docs	
	rm -rf chapters

