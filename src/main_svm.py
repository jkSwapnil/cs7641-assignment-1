# Main entrypoint for the SVM classifier

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
    # Boosting: kernel = "rbf"
    print("\nDataset: Rice | Algorithm: SVM")
    print("================================")
    print("Hyperparameters")
    print("---------------")
    print("- kernel = 'rbf'")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="rice",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="svm",
        kernel="rbf",
        probability=True,
        random_state=1693854383
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="rice", train_metrics=train_metrics)
    # Boosting: kernel = "poly", degree = 1
    print("\nDataset: Rice | Algorithm: SVM")
    print("================================")
    print("Hyperparameters")
    print("---------------")
    print("- kernel = 'poly'")
    print("- degree = 1")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="rice",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="svm",
        kernel="poly",
        degree=1,
        probability=True,
        random_state=1693854383
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="rice", train_metrics=train_metrics)


if execute_bank:
    # Bank Dataset
    # ------------
    # Boosting: kernel = "rbf"
    print("\nDataset: Bank | Algorithm: SVM")
    print("================================")
    print("Hyperparameters")
    print("---------------")
    print("- kernel = 'rbf'")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="bank",
        k=5,
        train_fracs=np.linspace(0.1, 1, 10),
        model_type="svm",
        kernel="rbf",
        probability=True,
        random_state=1693854383
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="bank", train_metrics=train_metrics)
    # Boosting: kernel = "poly", degree = 1
    print("\nDataset: Bank | Algorithm: SVM")
    print("================================")
    print("Hyperparameters")
    print("---------------")
    print("- kernel = 'poly'")
    print("- degree = 1")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="bank",
        k=5,
        train_fracs=np.linspace(0.1, 1, 10),
        model_type="svm",
        kernel="poly",
        degree = 1,
        probability=True,
        random_state=1693854383
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="bank", train_metrics=train_metrics)
