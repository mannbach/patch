# PATCH
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17177242.svg)](https://doi.org/10.5281/zenodo.17177242)

This project contains the code to produce, analyze and plot the networks created by `PATCH`, a network model of [P]referential [A]ttachment, [T]riadic [C]losure and [H]omophily.
This is an interface that simplifies the interaction with the [NetIn software package](https://cshvienna.github.io/NetworkInequalities/).
Check the linked [Zenodo page](https://doi.org/10.5281/zenodo.17160884) to download the intermediate results data.

## Requirements
To make this code independent of the underlying OS and software dependencies, we make us of docker compose (`>= v.2.12.2`, see [official docs](https://docs.docker.com/compose/) for installation steps).
Docker takes care of installing required software packages, and the setup of the local code.

## Configuration
Configuration of the docker builds is handled by `.env`-files.
An example of such a file can be found in `/config/sample.env`.
This file is sufficient to reproduce the findings.
It defines environment variables, such as folders for in- and output.

The high-level execution scripts in `/scripts/` additionally read configuration parameters from the command line.
If no argument is given, a default value is typically taken from `patch/constants.py`.
These scripts also provide a description of what they do when calling the `-h` flag, e.g.,
```bash
scripts/create_graphs.py -h
```

## Execution steps
### Building and running the docker image
After adjusting the [configuration](#configuration), build the docker image by running
```bash
docker compose --env-file config/sample.env build
```
replacing `config/sample.env` to the path to your custom `.env`-file in case you created a new one (from here on, we will continue to use `config/sample.env`).
Start the container by executing
```bash
docker compose --env-file config/sample.env up -d
```
with the `-d` flag signaling docker to run the containers in a background process, allowing you to continue using the same terminal.

### Running the code
All executable, high-level scripts are located in the `scripts/` folder.
To execute a script at location `<script.py>` and its arguments `<arguments>`, simply run (see below for examples):
```bash
docker exec -t patch python <script.py> <arguments>
```
The `-t` forwards the output of the container immediately to your local terminal.

Running the executable scripts in the presented order will reproduce the figures of the paper.
For instance, to compute a subset of graphs, run
```bash
docker exec -t patch python scripts/create_graphs.py -H .2 .5 -tau 0.0 0.2 -r 5 -lfm-g UNIFORM PAH -lfm-t UNIFORM
```

## The project
We explain the code structure by the folder tree
```
├── README.md  # this file
├── data  # folder containing data
│   ├── empirical  # empirical datasets
│   └── graphs  # folder containing raw graphs if present
├── notebooks  # jupyter notebooks
│   ├── plot_elfi_inference_results.ipynb  # empirical results plots
│   ├── plot_fig1_subplots.ipynb  # toy networks and connection probabilities
│   ├── plot_simulation_results.ipynb  # Simulation based results
│   └── plot_synthetic_inference_validation.ipynb  # inference validation SI plots
├── patch  # local project containing helper functions
│   ├── constants.py  # constant strings and values
│   ├── elfi.py  # package code to handle ELFI inference
│   ├── empirical.py  # processing of empirical datasets
│   ├── io.py  # functions for input/output
│   ├── model_config.py  # defines and validates the configuration of a model
│   ├── model_config.py  # defines and validates the configuration of a model
│   └── statistics.py  # aggregate network inequality statistics
├── output  # folder that contains output plots and stats
├── requirements.txt  # required software packages
├── scripts  # scripts to compute data
│   ├── create_graphs.py # creates graphs and stores them as JSON files
│   ├── elfi_inference.py # run ELFI inference on empirical datasets
│   ├── elfi_predictive.py # run ELFI predictive analysis
│   ├── elfi_validation.py # run ELFI synthetic validation analysis
│   └── compute_aggregate_stats.py # reads graphs and computes aggregate stats
└── setup.py
```

### output
The output folder contains aggregated network statistics (`stats/`), figures (`plots/`), and inference results (`inference/`).

Aggregated statistics are provided as a `stats/*.csv`-file with the following columns
- Parameters
    - `N`: Number of nodes.
    - `m`: Number of new links per node.
    - `lfm_global`: the global link formation mechanism. Values can be found in `utils/constants.py`.
    - `lfm_tc`the local link formation mechanism (for triadic closure links). Values can be found in `utils/constants.py`.
    - `homophily`: the homophily parameters. Higher `h` means that nodes prefer to connect to nodes of the same group.
    - `tau`: the triadic closure probability. Higher values mean that more links will be formed according to the `local` link formation mechanism.
    - `realization`: the realization (typically multiple simulation runs are executed).
- Network metrics
    - `gini`: the global degree inequality. Higher values indicate that degrees are distributed unequally: few nodes have most of the links.
    - `gini_min` and `gini_maj`: the degree inequality within each group.
    - `ei`: network segregation. Values towards `-1` indicate a segregated network in which nodes connect mainly to their own group. Values close to `+1` indicate the opposite.
    - `mann_whitney`: the inequity in terms of the minority. Values close to `+1` indicate that the minority generally has higher degrees than the majority group. Values towards `0.5` suggest equity and values towards `0.0` indicate majority advantage.

### data
The data folder contains the simulated and empirical networks (in sub-folders `graphs/` and `empirical/`).

The simulated graphs are stored as a zipped file (typically as `data/graphs/N-<N>_m-<m>_f-<f>.tar.gz`) with `<N>`, `<m>`, and `<f>` being placeholders for the respective variable values.
Run ```tar -xf data/graphs/N-<N>_m-<m>_f-<f>.tar.gz``` to extract the graph files.
Each graph is contained in a [JSON](https://de.wikipedia.org/wiki/JavaScript_Object_Notation) generated by NetworkX (see [docs](https://networkx.org/documentation/stable/reference/readwrite/json_graph.html) for details and [this link](https://gist.github.com/mbostock/4062045) for an example using D3).
The parameters used to create each graphs can be found in the file name or as values in the respective files.
The structure of each JSON file is explained by the example of `data/graphs/N-5000_m-3_f-0.2/N-5000_m-3_f-0.2_h-0.25_tau-0.0_lfm-g-HOMOPHILY_lfm-t-HOMOPHILY_r-0.json`:

```json
{
    "N": 5000,
    "m": 3,
    "f_m": 0.2,
    "tau": 0.0,
    "lfm_global": "HOMOPHILY",
    "lfm_tc": "HOMOPHILY",
    "realization": 0,
    "h_M": 0.25,
    "h_m": 0.25,
    "minority": [...],
    "edge_list": [...]
}
```
with `"minority"` containing minority node IDs and `"edge_list"` containing tuples `(u,v)`of connected nodes.
See [notebooks](#notebooks) for examples how to load these datasets and [scripts](#scripts) to create your own data.

### notebooks
Contains notebooks that create the plots of the paper.
You can run a local jupyter server by running ```jupyter notebook``` in your terminal and the open the presented weblink in your browser to run the notebooks (see [documentation](https://docs.jupyter.org/en/latest/) for details).

### patch
This folder has general functions that simplify the interaction with the created data and the [NetIn package](https://cshvienna.github.io/NetworkInequalities/).
They can be imported in an arbitrary Python script as (see [notebooks](#notebooks) or [scripts](#scripts) for examples):
```python
from patch.model_config import ModelConfig
```

### scripts
Contains Python scripts to simulate networks, compute the aggregate statistics required to produces the presented plots and model inference using the [elfi](https://elfi.readthedocs.io/en/latest/) software package.
The aggregated statistics will also be provided as a `.csv`-file in [Zenodo](https://doi.org/10.5281/zenodo.17160884).

## Contact
In case of questions or bugs, open an issue in [github](https://github.com/mannbach/patch/issues) or contact me directly at `bachmann \at csh.ac.at`.
