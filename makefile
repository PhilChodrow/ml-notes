.ONESHELL: 

SHELL = /bin/zsh
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate


stage: 
	$(CONDA_ACTIVATE) ml-0451
	python scripts/prep-qmd.py
	quarto render --profile publish
	python scripts/create-ipynb.py
	python scripts/insert-colab-link.py

publish: 
	$(CONDA_ACTIVATE) ml-0451
	python scripts/prep-qmd.py
	quarto render --profile publish
	python scripts/create-ipynb.py
	python scripts/insert-colab-link.py
	git add .
	git commit -m "Update"
	git push

prep: 
	$(CONDA_ACTIVATE) ml-0451
	python scripts/create-ipynb.py
	python scripts/prep-qmd.py

preview: 
	$(CONDA_ACTIVATE) ml-0451
	quarto preview --profile preview

clean: 
	find . -type f -name "* [0-9]*" -delete
	find . -name "* [0-9]*" -type d -exec rm -r "{}" \;
	rm -rf docs	
	rm -rf chapters

