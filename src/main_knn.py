# Main entrypoint for the KNN classifier

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
    # KNN classifier: n_neighbors = 30, p = 2
    print("\nDataset: Rice | Algorithm: KNN")
    print("================================")
    print("Hyperparameters")
    print("---------------")
    print("- n_neighbors = 30")
    print("- p = 2")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="rice",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="knn",
        n_neighbors=30,
        p=2
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="rice", train_metrics=train_metrics)
    # KNN classifier: n_neighbors = 120, p = 2
    print("\nDataset: Rice | Algorithm: KNN")
    print("================================")
    print("Hyperparameters")
    print("---------------")
    print("- n_neighbors = 120")
    print("- p = 2")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="rice",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="knn",
        n_neighbors=120,
        p=2
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="rice", train_metrics=train_metrics)
    # KNN classifier: n_neighbors = 200, p = 2
    print("\nDataset: Rice | Algorithm: KNN")
    print("================================")
    print("Hyperparameters")
    print("---------------")
    print("- n_neighbors = 200")
    print("- p = 2")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="rice",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="knn",
        n_neighbors=200,
        p=2
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="rice", train_metrics=train_metrics)
    # KNN classifier: n_neighbors = 200, p = 1
    print("\nDataset: Rice | Algorithm: KNN")
    print("================================")
    print("Hyperparameters")
    print("---------------")
    print("- n_neighbors = 200")
    print("- p = 1")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="rice",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="knn",
        n_neighbors=200,
        p=1
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="rice", train_metrics=train_metrics)
    print()

if execute_bank:
    # Bank Dataset
    # ------------
    # KNN classifier: n_neighbors = 30, p = 2
    print("\nDataset: Bank | Algorithm: KNN")
    print("================================")
    print("Hyperparameters")
    print("---------------")
    print("- n_neighbors = 30")
    print("- p = 2")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="bank",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="knn",
        n_neighbors=30,
        p=2
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="bank", train_metrics=train_metrics)
    # KNN classifier: n_neighbors = 120, p = 2
    print("\nDataset: Bank | Algorithm: KNN")
    print("================================")
    print("Hyperparameters")
    print("---------------")
    print("- n_neighbors = 120")
    print("- p = 2")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="bank",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="knn",
        n_neighbors=120,
        p=2
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="bank", train_metrics=train_metrics)
    # KNN classifier: n_neighbors = 200, p = 2
    print("\nDataset: Bank | Algorithm: KNN")
    print("================================")
    print("Hyperparameters")
    print("---------------")
    print("- n_neighbors = 200")
    print("- p = 2")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="bank",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="knn",
        n_neighbors=200,
        p=2
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="bank", train_metrics=train_metrics)
    # KNN classifier: n_neighbors = 200, p = 1
    print("\nDataset: Bank | Algorithm: KNN")
    print("================================")
    print("Hyperparameters")
    print("---------------")
    print("- n_neighbors = 200")
    print("- p = 1")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="bank",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="knn",
        n_neighbors=200,
        p=1
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="bank", train_metrics=train_metrics)
