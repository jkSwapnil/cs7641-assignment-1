# Main entrypoint for the decision tree classifier

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
    # Decision tree classifier: min_samples_leaf = 1, ccp_alpha = 0.002
    print("\nDataset: Rice | Algorithm: Decision Tree")
    print("========================================")
    print("Hyperparameters")
    print("---------------")
    print("- min_samples_leaf = 1")
    print("- ccp_alpha = 0.002")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="rice",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="decision_tree",
        min_samples_leaf=1,
        ccp_alpha=0.002,
        random_state=1693854383,
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="rice", train_metrics=train_metrics)
    # Decision tree classifier: min_samples_leaf = 1, ccp_alpha = 0.01
    print("\nDataset: Rice | Algorithm: Decision Tree")
    print("========================================")
    print("Hyperparameters")
    print("---------------")
    print("- min_samples_leaf = 1")
    print("- ccp_alpha = 0.01")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="rice",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="decision_tree",
        min_samples_leaf=1,
        ccp_alpha=0.01,
        random_state=1693854383,
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="rice", train_metrics=train_metrics)
    # Decision tree classifier: min_samples_leaf = 20, ccp_alpha = 0.01
    print("\nDataset: Rice | Algorithm: Decision Tree")
    print("========================================")
    print("Hyperparameters")
    print("---------------")
    print("- min_samples_leaf = 20")
    print("- ccp_alpha = 0.01")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="rice",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="decision_tree",
        min_samples_leaf=20,
        ccp_alpha=0.01,
        random_state=1693854383,
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="rice", train_metrics=train_metrics)
    print()


if execute_bank:
    # Bank Dataset
    # ------------
    # Decision tree classifier: min_samples_leaf = 1, ccp_alpha = 0.005
    print("\nDataset: Bank | Algorithm: Decision Tree")
    print("========================================")
    print("Hyperparameters")
    print("---------------")
    print("- min_samples_leaf = 1")
    print("- ccp_alpha = 0.005")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="bank",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="decision_tree",
        min_samples_leaf=1,
        ccp_alpha=0.005,
        random_state=1693854383,
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="bank", train_metrics=train_metrics)
    # Decision tree classifier: min_samples_leaf = 1, ccp_alpha = 0.0001
    print("\nDataset: Bank | Algorithm: Decision Tree")
    print("========================================")
    print("Hyperparameters")
    print("---------------")
    print("- min_samples_leaf = 1")
    print("- ccp_alpha = 0.0001")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="bank",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="decision_tree",
        min_samples_leaf=1,
        ccp_alpha=0.0001,
        random_state=1693854383,
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="bank", train_metrics=train_metrics)
    # Decision tree classifier: min_samples_leaf = 20, ccp_alpha = 0.005
    print("\nDataset: Bank | Algorithm: Decision Tree")
    print("========================================")
    print("Hyperparameters")
    print("---------------")
    print("- min_samples_leaf = 20")
    print("- ccp_alpha = 0.005")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="bank",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="decision_tree",
        min_samples_leaf=20,
        ccp_alpha=0.005,
        random_state=1693854383,
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="bank", train_metrics=train_metrics)
    print()
