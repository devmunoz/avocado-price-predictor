# Avocado Price Predictor 🥑🤑

###### Last update: 12/2024

#### Machine Learning Project

## Description

This repository contains a hyperparameter processor to brute-force ML model training and select the best model based on the metrics.

The source data was extracted from the [Hass Avocado Board](https://hassavocadoboard.com/) website. The EDA and ETL processes have been omitted from this repository.

This repository is part of the original project carried out by [Patricia G-R Palombi](https://www.linkedin.com/in/patricia-g-r-palombi-269b78183/), [José Dos Reis - josedosr](https://github.com/josedosr), [Pamela Colman - pamve](https://github.com/pamve), and myself. If you want more information, please check the original project publication on LinkedIn [here](https://www.linkedin.com/posts/dmunoz-m_proyecto-02-avocado-temporal-series-activity-7165847101237542912-lvkd?utm_source=share&utm_medium=member_desktop).

## Installation and Execution

- **Prerequisites**:

  - Install [Python](https://www.python.org/downloads/) and [Virtual Environment (venv)](https://docs.python.org/3/library/venv.html) on your machine.
  - Clone this repository.
  - [optional] Install [Jupyter Lab](https://jupyter.org/install).

- **Run the scripts**:

  - Install the virtual environment:

    ```
    python -m venv .venv
    ```
  - Activate the virtual environment:

    ```
    source .venv/bin/activate
    ```
  - Install the requirements:

    ```
    pip install -r requirements.txt
    ```
  - **Usage** 😄:

    ```
    usage: hyperparams_model_processer.py [-h] [-e] [-p]

    Hyperparameter models executor

    options:
      -h, --help     show this help message and exit
      -e, --execute  Build and execute models. WARNING! THIS CAN BE A VERY HEAVY PROCESS
      -p, --plot     Plot models' performance results
    ```
  - **ALTERNATIVE:** Open the [.ipynb notebook](hyperparam_models_processer.ipynb) and just follow the content.

## Contribution

Feel free to improve or update the code.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

