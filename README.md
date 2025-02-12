# Perspectives on Intelligence - Community Survey

Analysis of the results of the Perspectives on Intelligence Community Survey. The term ‘intelligence’ is often used in various fields including Artificial Intelligence, Cognitive Science, Natural Language Processing, and Machine Learning. But research papers rarely specify what is meant by that term. As a result, we as a research community often talk past each other, even about our basic premises and goals. Authors may even change how they write their papers, based on a wrong impression of what most other researchers believe.

For more information on the project see the [Project Page](https://bertramhojer.github.io/projects/intelligence-survey/)!

## Replication

If you wish to run the analysis we have defined the following commands that can be run using uv. First install all dependencies by running:
```
uv sync
```

### Generating analysis report
```
uv run generate-report
```

### Generating visualisations from the [Paper](https://bertramhojer.github.io/projects/intelligence-survey/). 
```
uv run generate-figures
```

Be aware to rerun the analysis you need to download the curated dataset. Contact the first author (berh)at(itu).(dk) to get access to the curated survey responses.


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
│   │   ├── analysis.py       # analysis script
│   │   ├── data_mappings.py  # map for processing
│   │   ├── data.py           # data processing
│   │   ├── demo_mappings.py  # map for processing
│   │   ├── report.py         # generate summary report
│   │   ├── visualize.ipynb   # visualization notebook
│   │   └── visualize.py      # visualization script
├── .gitignore
├── pyproject.toml            # Python project file
└── README.md                 # Project README
```
