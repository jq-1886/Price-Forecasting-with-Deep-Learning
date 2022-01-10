<div align="center">
	<h1>ACSE-9 Independent Research Project</h1>
    	<h2>Financial Security Price Prediction with Deep Learning</h2>
      	<h3>Jack Quested</h3>
	<h3>CID: 01123431 // Github ID: acse-jaq15</h3>
	<h3>Applied Computational Science and Engineering MSc</h3>
	<h3>Imperial College London</h3>
	<h4>This repository contains the code, notebooks and report created for a comparative study in various Deep Learning architectures and their ability to perform univariate time series forecasting</h4>
</div>

***

# Contents
- [Overview](#Overview)
- [Languages](#Languages)
- [data](#Data)
- [evaluation_notebooks](#evaluation_notebooks)
- [example_notebooks](#example_notebooks)
- [group_plots](#group_plots)
- [metrics](#metrics)
- [metrics_extended](#metrics_extended)
- [model_graphs](#model_graphs)
- [model_graphs_extended](#model_graphs_extended)
- [multi_plots](#multi_plots)
- [notebooks](#notebooks)
- [output_df](#output_df)
- [output_df_extended](#output_df_extended)
- [output_excel](#output_excel)
- [output_excel_extended](#output_excel_extended)
- [plots](#plots)
- [plots_extended](#plots_extended)
- [saved_models](#saved_models)
- [saved_models_extended](#saved_models_extended)
- [tests](#tests)
- [LICENSE](#LICENSE)
- [base_model.py](#base_model.py)
- [best_configs.txt](#best_configs.txt) 
- [data_reader.py](#data_reader.py)
- [extract_configs.py](#extract_configs.py)
- [model_loader.py](#model_loader.py)
- [move_json.py](#move_json.py)
- [requirements.txt](#requirements.txt)
- [security_plotter.py](#security_plotter.py)

***

# Overview
An explanation of the purpose and contents of all directories in this repo are contained in the relevant sections. This project focuses on univariate time series prediction, where the only variable on which price predictions are made is the price of the security itself.

All models use a 30 day window of preceeding closing prices to then give a prediction 1 day ahead, the next days closing price.

Hyperparameter searching was conducted using [Weights and Biases](http://www.wandb.ai/), where each model was optimised for each security, resulting in more than 5000 possible model configurations being attempted. These searches were performed in Google Colaboratory notebooks with Weights and Biases integration.

The most optimal configurations were then tested on the test dataset. For each security, the best performing model was then adapted to handle a greater predcition horizon of 15 days as an extenstion of the study.

Also note, notebooks in `./notebooks` directory rely on a [feeder repo](https://github.com/acse-jaq15/feeder_repo) in order to load necessary modules and data. The reason for this is to allow for easy replication of notebooks on different machines and GitHub user accounts due to the limtations and privacy concerns of creating notebooks that have private SSH keys saved in them. For this reason feeder repo is a public repo while this repo is private, while the contents of the feeder repo are copied from the latest version of this repo.

Models used:
- Convolutional Neural Network (CNN):
  - 'CNN' - vanilla CNN
  - 'CNN_GRU' - CNN with GRU units
  - 'CNN_LSTM' - CNN with LSTM units

- Gated Recurrent Unit Network (GRU):
  - 'GRU' - stacked GRU
  - 'GRU_AE' - stacked GRU autoencoder
  - 'GRU_LSTM' - stacked GRU LSTM hybrid

- Long Short-Term Memory Network (LSTM):
  - 'LSTM' - stacked LSTM
  - 'LSTM_AE' - stacked LSTM autoencoder
  - 'LSTM_GRU' - stacked LSTM GRU hybrid

- Multi-Layer Perceptron (MLP):
  - 'MLP' - Multi-layer perceptron
  - 'MLP_AE' - Multi-layer perceptron autoencoder

Securities used:
- Commodities:
  - 'Al' - LME 3 month aluminium futures
  - 'Cu' - LME 3 month copper futures
  - 'Corn' - CME rolling active month corn futures

- Currencies:
  - 'EURCHF' - spot Euro / Swiss currency pair
  - 'EURUSD' - spot Euro / US dollar currency pair
  - 'GBPUSD' - spot British pound / US dollar currency pair

- Fixed Income:
  - 'Bund10y' - rolling 10y German Bund (yield)
  - 'Gilt10y' - rolling 10y British Gilt (yield)
  - 'Treasury10y' - rolling 10y US Treasury (yield)

- Equities:
  - 'Amazon' - NASDQ listed Amazon.com Inc. common stock
  - 'Google' - NASDAQ listed Alphabet Inc. class A common stock
  - 'Nvidia' - NASDAQ listed NVIDIA Corporation common stock

The results of the hyperparmeter tuning via [Weights and Biases](http://www.wandb.ai/) can be visualised with these links:
- [CNN](https://wandb.ai/acse-jaq15/ACSE_9_CNN?workspace=user-acse-jaq15)
- [CNN_GRU](https://wandb.ai/acse-jaq15/ACSE_9_CNN_GRU?workspace=user-acse-jaq15)
- [CNN_LSTM](https://wandb.ai/acse-jaq15/ACSE_9_CNN_LSTM?workspace=user-acse-jaq15)
- [GRU](https://wandb.ai/acse-jaq15/ACSE_9_GRU?workspace=user-acse-jaq15)
- [GRU_AE](https://wandb.ai/acse-jaq15/ACSE_9_GRU_AE?workspace=user-acse-jaq15)
- [GRU_LSTM](https://wandb.ai/acse-jaq15/ACSE_9_GRU_LSTM?workspace=user-acse-jaq15)
- [LSTM](https://wandb.ai/acse-jaq15/ACSE_9_LSTM?workspace=user-acse-jaq15)
- [LSTM_AE](https://wandb.ai/acse-jaq15/ACSE_9_LSTM_AE?workspace=user-acse-jaq15)
- [LSTM_GRU](https://wandb.ai/acse-jaq15/ACSE_9_GRU_LSTM?workspace=user-acse-jaq15)
- [MLP](https://wandb.ai/acse-jaq15/ACSE_9_MLP?workspace=user-acse-jaq15)
- [MLP_AE](https://wandb.ai/acse-jaq15/ACSE_9_MLP_AE?workspace=user-acse-jaq15)

***

# Languages
- Language used to build modules and notebooks is Python 3.2.10.
- It is recommended that the user installs requirements [requirements.txt](#requirements.txt) before attempting to build to use any modules contained in this repo via `pip install requirements.txt`.

***

# data
- Directory contains time series data for each security used in the study.
- Data is stored in `.csv` format, with a date range from the 1st trading day in 2014 to the last trading day in 2019.
- The number of trading days in the date range differs for each security due to the number of public holidays in that securities listing location and exchange practices around end of year trading days.
- The [data_reader.py](#data_reader.py) module handles the extraction of validation, training and test datasets, please see that section for more information.

***

# evaluation_notebooks
- Directory contains Colab notebooks used to evaluate the models, record error metrics and generate plots of predictions.
- Evaluations is performed through calls to `Trained_Model` class of [model_loader.py](#model_loader.py) module, please see relevant section for more information.
- Results are written to a Pandas DataFrame and is stored in [output_df](#output_df) directory as a .pkl file.

***

# example_notebooks
- Directory contains Colab notebooks that demonstrate possible implementations of `Trained_Model` and `Untrained_Model` classes of [model_loader.py](#model_loader.py) module in order to load the saved or default models used in this study.
- Loading trained models [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/acse-2020/acse2020-acse9-finalreport-acse-jaq15/blob/main/example_notebooks/Trained_Example.ipynb)
- Loading untrained models [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/acse-2020/acse2020-acse9-finalreport-acse-jaq15/blob/main/example_notebooks/Untrained_Example.ipynb)

***

# group_plots
- Directory contains plots grouped by security, where each plot charts the predictions of each model for that security.

***

# metrics
- Directory contains the error metrics recorded by each model for each security as .txt files.

***

# metrics_extended
- Directory contains the error metrics recorded by the models chosen to output an extended prediction horizon as .txt. files.

***

# model_graphs
- Directory contains graphs of each model used for each security as .png files.
- Generated by calls to `tf.keras.utils.plot_model`.
- Files are outputs of relevant notebooks in `./notebooks/`.

***

# multi_plots
- Directory contains plots for each security, where the model prediction is recorded on separate subplots.
- Files are outputs of `./evaluation_notebooks/Model_Evaluation_and_Multi_Plotting.ipynb` notebook, linked here [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/acse-2020/acse2020-acse9-finalreport-acse-jaq15/blob/main/evaluation_notebooks/Model_Evaluation_and_Multi_Plotting.ipynb)

***

# notebooks
- Directory contains folders for each model, where each model folder contains 12 notebooks, corresponding to each security.
- These notebooks are Weights and Biases integrated and are where the models are implemnted, hyperparameter tuned and trained, tested and finally saved.

***

# output_df
- Directory contains a Pandas DataFrame saved as a `.pkl` file. DataFrame contains saved model error metrics and configurations.
- This file is an output of `./evaluation_notebooks/Model_Evaluation_and_Multi_Plotting.ipynb` notebook, linked here [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/acse-2020/acse2020-acse9-finalreport-acse-jaq15/blob/main/evaluation_notebooks/Model_Evaluation_and_Multi_Plotting.ipynb)

*** 

# output_df_extended
- Directory contains a Pandas DataFrame saved as a `.pkl` file. DataFrame contains extended model error metrics.
- This file is an output of `./evaluation_notebooks/Extended_Model_Predictions.ipynb` notebook, linked here [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/acse-2020/acse2020-acse9-finalreport-acse-jaq15/blob/main/evaluation_notebooks/Extended_Model_Predictions.ipynb)

***

# output_excel
- Directory contains an excel spreadsheet saved as a `.xlsx` file. Excel file contains model error metrics and configurations.
- This file is an output of `./evaluation_notebooks/Model_Evaluation_and_Multi_Plotting.ipynb` notebook, linked here [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/acse-2020/acse2020-acse9-finalreport-acse-jaq15/blob/main/evaluation_notebooks/Model_Evaluation_and_Multi_Plotting.ipynb)

***

# output_excel_extended
- Directory contains an excel spreadsheet saved as a `.xlsx` file. Excel file contains extended model error metrics.
- This file is an output of `./evaluation_notebooks/Extended_Model_Predictions.ipynb` notebook, linked here [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/acse-2020/acse2020-acse9-finalreport-acse-jaq15/blob/main/evaluation_notebooks/Extended_Model_Predictions.ipynb)

***

# plots
- Directory contains `.png` images of single model and security combination plots of actual, predicted and dummy prices.
- Files are outputs of the relevant notebooks in `./notebooks/`.

***

# plots_extended
- Directory contains `.png` images of extended single model and security combination plots of actual, predicted and dummy prices.
- Files are outputs of `./evaluation_notebooks/Extended_Model_Predictions.ipynb` notebook, linked here [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/acse-2020/acse2020-acse9-finalreport-acse-jaq15/blob/main/evaluation_notebooks/Extended_Model_Predictions.ipynb)

***

# saved_models
- Directory contains saved Keras models. These models are the final models used in this study.
- Models are outputs of the relevant notebooks in `./notebooks/`, saved via calls to `keras.models.save_model()` function.

***

# saved_models_extended
- Directory contains extended saved Keras models. These models are the models used with extended prediction horizons.
- Models are outputs of outputs of `./evaluation_notebooks/Extended_Model_Predictions.ipynb` notebook, linked here [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/acse-2020/acse2020-acse9-finalreport-acse-jaq15/blob/main/evaluation_notebooks/Extended_Model_Predictions.ipynb)

***

# tests
- Directory contains unit and CI tests.

***

# LICENSE
- MIT License.

***

# base_model.py
- Module contains `Base_Model` class, used to implement a base or dummy model.
- `Base_Model` makes a prediction that is simply the mean of 30 day window.
- Please see documentation within module for more information.

***

# best_configs.txt
- File contains best configurations used by final saved models following hyperparameter searches.
- Configurations are stored as a single dictionary written in string form, allowing for easy loading using either `json` or `ast` python libraries.
- Parameters included are:
  - Number of epochs model has been trained for.
  - Activation function used by model.
  - Optimizer used by model.
  - Optimizer learning rate.
  - Optimzier batch size.
- Note these configurations are the same configurations that led to the saved models that can be loaded via calls to `Trained_Model` class of `model_loader.py` module.
- File is an output of `extract_configs.py` script, see relevant section for more details.

***

# data_reader.py
- Module contains `Data_Reader` class, used to handle data.
- `Data_Reader` imports data from relevant `.csv` files in `./data/` directory, then allows user to perform:
  - Normalisation using 0,1 min/max scaling.
  - Extraction of training, test and validation datasets.
  - Extraction of relevant X and y values.
  - Extraction of real un-normalised prices.
- Please see documentation within module for more information.

***

# extract_configs.py
- File contains a Python script to extract best configurations from notebooks contained in `./notebooks` directory.
- Script generates `best_configs.txt` file, saving configurations as a dictionary to allow for easy extraction with `json` or `ast` Python libraries.
- Note, script requires notebook data to be converted to `.json` files to be read. These files have not been included but user can generate them by running `move_json.py` script.

***

# model_loader.py
- Module contains `Trained_Model` and `Untrained_Model` classes, used to load saved or default models.
- `Trained_Model` loads saved final models generated by notebooks in `./notebooks/` directory, then allows user to perform:
  - Additional training of model for a specified number of epochs.
  - Evaluation of loaded model, either before or after additional training, error metrics used are Mean Squared Error, Root Mean Squared Error and Mean Absolute Error.
- `Trained_Model` is parent class to `Untrained_Model`.

- `Untrained_Model` loads untrained model, model has default configuration, then allows user to perform:
  - Training of model for a specified number of epochs.
  - Evaluation of loaded model, error metrics used are Mean Squared Error, Root Mean Squared Error and Mean Absolute Error.
  - `Untrained_Model` is child class of `Trained_Model` and inherits `Train()` and `Evaluate()` methods.

- Please see documentation within modules for more information.

***

# move_json.py
- File contains a Python script to convert notebooks contained in `./notebooks/` directory from `.ipynb` format to `.json` format.
- Script must be run before `extract_configs.py` is used.
- Note, script is simply intended to allow for easy access of notebook `json` data in order to compile dictionary of best model configurations.

***

# requirements.txt
- File contains repo requirements stored as `.txt` file.
- Please run before using modules via `pip install requirements.txt`.

***

# security_plotter.py
- Module contains `Security_Plotter` class, used to plot single model security combinations, plotting actual, predicted and dummy prices.
- Please see documentation within module for more information.

***
