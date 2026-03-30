# Human Capital Expenditure and Its Effectiveness on Multi-Programme Coverage: A Policy Prioritization Investigation [data and code repository]

### Authors: Omar A. Guerrero<sup>1</sup>, Daniele Guariso<sup>2</sup>, Gonzalo Castañeda<sup>3</sup>, and Michael Weber<sup>4</sup>

<sup>1</sup> University of Helsinki
<sup>2</sup> Euro-Mediterranean Center on Climate Change
<sup>3</sup> Centro de Investigación y Docencia Económicas (CIDE), Mexico City
<sup>4</sup> The World Bank


## Description
This repository contains the code and data to replicate the analysis performed in the Workd Bank Working Paper titled: *Human Capital Expenditure and Its Effectiveness on Multi-Programme Coverage: A Policy Prioritization Investigation*.
The analysis makes use of the <a href="https://policypriority.org" target="_blank">Policy Priority Inference (PPI)</a> framework to model connections between government expenditure and target indicator coverage in Mexico between 2016 and 2022.
The paper presents various types of analysis that aim at identifying potential opportunities and challenges in coordinating expenditure strategies that aim at improving Mexico's stock of human capital.


## Paper abstract
*This paper uses AI-enhanced agent computing to determine how to allocate budgetary resources within a large set of heterogeneous government programs targeting human capital. Our approach considers essential features of the budget allocation process: multidimensionality, interdependencies between policy issues, and the political economy of public officials' collective action. We use highly disaggregated Mexican data covering the 2016-2022 period across 49 human capital programs of the federal government and focus on how expenditure affects program coverage (the proportion of the population with a public problem and who has access to various government benefits to mitigate those problems) in the short run. We answer the following research questions: how sensitive is program coverage to changes in public expenditure?; what are the structural bottlenecks behind poor coverage response?; and what are the optimal budgetary allocations that could boost the performance of a multidimensional objective function?*


## The structure
The repository is organized into three folders:
- `code`: contains all the scripts needed to process the raw datasets and perform the simulations for the analysis
- `data`: provides all the necessary data
- `figures`: provides high resolution files of all the figures in the report
- `tables`: contains the tables from the report


## The code
The code is organized into sequential Python scripts. They should be run in the order indicated by the number in the filenames.
Files from 1 to 10 are for processing data.
Files from 11 to 20 calibrate the PPI model and run all the experiments.
Scripts 21 onwards produce all the figures in the report and saves them in the `figures` folder.
Once all files have been run, the user can take the output data files and replicate the figures presented in the report.


## Policy Priority Inference (PPI)
The analysis requires the <a href="https://policypriority.org" target="_blank">PPI</a> toolkit, which can be <a href="https://pypi.org/project/policy-priority-inference/" target="_blank">installed for Python through pypi</a>. Further information on PPI can be found in the book: <a href="https://www.cambridge.org/core/books/complexity-economics-and-sustainable-development/BD6CCB51DF29A5FE3638B3B99C7D0CB1" target="_blank">Complexity Economics and Sustainable Development</a>. The open source-code for the PPI framework can be found in its <a href="https://github.com/oguerrer/ppi" target="_blank">official repository</a>.


## How to use
If all the necessary libraries have been properly installed, all needed is cloning this repository (preserving the folder structure) and run each script sequentially.
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









