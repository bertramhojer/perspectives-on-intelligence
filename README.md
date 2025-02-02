# Perspectives on Intelligence - Community Survey

Analysis of the results of the Perspectives on Intelligence Community Survey. The term ‘intelligence’ is often used in various fields including Artificial Intelligence, Cognitive Science, Natural Language Processing, and Machine Learning. But research papers rarely specify what is meant by that term. As a result, we as a research community often talk past each other, even about our basic premises and goals. Authors may even change how they write their papers, based on a wrong impression of what most other researchers believe.

For more information on the project see the [Project Page](https://bertramhojer.github.io/projects/intelligence-survey/)!

## Project structure

The directory structure of the project looks like this:
```txt
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── notebooks/                # Jupyter notebooks
├── figures/                  # Figures
├── src/                      # Source code
│   ├── survey/
│   │   ├── __init__.py
│   │   ├── analyzer.py       # analysis script
│   │   ├── data_mappings.py
│   │   ├── data.py           # data preprocessing
│   │   ├── demo_mappings.py
│   │   ├── report.py         # generate summary report
│   │   └── visualize.py      # visualization script
├── .gitignore
├── pyproject.toml            # Python project file
└── README.md                 # Project README
```
