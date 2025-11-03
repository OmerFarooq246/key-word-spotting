conda env:
1. eval "$(/Users/traveler/anaconda3/bin/conda shell.zsh hook)" 
2. conda activate /Users/traveler/Desktop/VSCODEs/research/venv
3. conda env export > venv.yml | conda env create -f environment.yml

installs:
1. pip install pydub