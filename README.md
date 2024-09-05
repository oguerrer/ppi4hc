# Human Capital Expenditure and Its Effectiveness on Multi-Programme Coverage: A Policy Priority Inference Investigation

This repository contains the code and data to replicate the analysis performed in the report with the same title.

## The structure
The repository is organized into three folders:
- `code`: contains all the scripts needed to process the raw datasets and perform the simulations for the analysis
- `data`: provides all the necessary data
- `figures`: provides high resolution files of all the figures in the report 

## The code
The code is organized into sequential Python scripts. They should be run in the order indicated by the number in the filenames.
Files from 1 to 10 are for processing data.
Files from 11 to 20 calibrate the PPI model and run all the experiments.
Scripts 21 onwards produce all the figures in the report and saves them in the `figures` folder.
Once all files have been run, the user can take the output data files and replicate the figures presented in the report.

## Policy Priority Inference
The analysis requires the <a href="https://policypriority.org" target="_blank">Policy Priority Inference (PPI)</a> toolkit, which can be <a href="https://pypi.org/project/policy-priority-inference/" target="_blank">installed for Python through Pypi</a>. Further information on PPI can be found in the book: <a href="https://www.cambridge.org/core/books/complexity-economics-and-sustainable-development/BD6CCB51DF29A5FE3638B3B99C7D0CB1" target="_blank">Complexity Economics and Sustainable Development</a>.

## How to use
If all the necessary libraries have been properly installed, all needed is to clone this repository (preserving the folder structure) and run each script sequentially.
The repository already provides all the intermediate data files, so it is not necessary to run every single script.
For example, if you want to modify the experiments from section 4.1 (sensitivity analysis), you can go straight to script 12, modify it, and run it.

## Data sources
The raw data files come from the following sources:

- <a href="https://www.coneval.org.mx/evaluacion/ipfe/Paginas/default.aspx" target="_blank">CONEVAL's federal inventory of social programs (contains data on both expenditure and performance)</a>
- <a href="https://www.worldbank.org/en/publication/worldwide-governance-indicators" target="_blank">The World Bank's worldwide governance indicators</a>
- <a href="https://prosperitydata360.worldbank.org/en/dataset/IMF+CPI" target="_blank">The World Bank's consumer price index database</a>
- <a href="https://data.worldbank.org/indicator/SP.POP.TOTL" target="_blank">The World Bank's total population index database</a>
- <a href="https://data.worldbank.org/indicator/HD.HCI.OVRL?cid=GGH_e_hcpexternal_en_ext" target="_blank">The World Bank's human capital index database</a>
- <a href="https://www.rug.nl/ggdc/productivity/pwt/?lang=en" target="_blank">Penn world tables</a>


