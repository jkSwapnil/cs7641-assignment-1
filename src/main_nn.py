# Main entrypoint for the neural network classifier

import sys

import numpy as np

from train import Trainer
from test import Tester
from plots import PlotTrainMetrices

# Weather to execute for rice and bank dataset
execute_rice = True
execute_bank = True
if len(sys.argv) == 2:
    if sys.argv[1] == "rice":
        execute_bank = False
    elif sys.argv[1] == "bank":
        execute_rice = False
    else:
        print(f"Warning: '{sys.argv[1]}' dataset is not supported. Reverting to default behaviour.")

if execute_rice:
    # Rice Dataset
    # ------------
    # Neural network: hidden_layer_sizes = [50, 50], activation = 'relu'
    print("\nDataset: Rice | Algorithm: Neural network")
    print("===========================================")
    print("Hyperparameters")
    print("---------------")
    print("- hidden_layer_sizes = [50, 50]")
    print("- activation = 'relu'")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="rice",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="neural_network",
        hidden_layer_sizes=[50, 50],
        activation = "relu",
        random_state=1693854383
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="rice", train_metrics=train_metrics)
    # Neural network: hidden_layer_sizes = [100], activation = 'relu'
    print("\nDataset: Rice | Algorithm: Neural network")
    print("===========================================")
    print("Hyperparameters")
    print("---------------")
    print("- hidden_layer_sizes = [100]")
    print("- activation = 'relu'")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="rice",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="neural_network",
        hidden_layer_sizes=[100],
        activation = "relu",
        random_state=1693854383
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="rice", train_metrics=train_metrics)
    # Neural network: hidden_layer_sizes = [100], activation = 'logistic'
    print("\nDataset: Rice | Algorithm: Neural network")
    print("===========================================")
    print("Hyperparameters")
    print("---------------")
    print("- hidden_layer_sizes = [100]")
    print("- activation = 'logistic'")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="rice",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="neural_network",
        hidden_layer_sizes=[100],
        activation = "logistic",
        random_state=1693854383
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="rice", train_metrics=train_metrics)


if execute_bank:
    # Bank Dataset
    # ------------
    # Neural network: hidden_layer_sizes = [50, 50], activation = 'relu'
    print("\nDataset: Bank | Algorithm: Neural network")
    print("===========================================")
    print("Hyperparameters")
    print("---------------")
    print("- hidden_layer_sizes = [50, 50]")
    print("- activation = 'relu'")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="bank",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="neural_network",
        hidden_layer_sizes=[50, 50],
        activation="relu",
        random_state=1693854383,
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="bank", train_metrics=train_metrics)
    # Neural network: hidden_layer_sizes = [100], activation = 'relu'
    print("\nDataset: Bank | Algorithm: Neural network")
    print("===========================================")
    print("Hyperparameters")
    print("---------------")
    print("- hidden_layer_sizes = [100]")
    print("- activation = 'relu'")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="bank",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="neural_network",
        hidden_layer_sizes=[100],
        activation="relu",
        random_state=1693854383,
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="bank", train_metrics=train_metrics)
    # Neural network: hidden_layer_sizes = [100], activation = 'logistic'
    print("\nDataset: Bank | Algorithm: Neural network")
    print("===========================================")
    print("Hyperparameters")
    print("---------------")
    print("- hidden_layer_sizes = [100]")
    print("- activation = 'logistic'")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="bank",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="neural_network",
        hidden_layer_sizes=[100],
        activation="logistic",
        random_state=1693854383,
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="bank", train_metrics=train_metrics)
