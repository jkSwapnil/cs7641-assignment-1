# Main entrypoint for boosting tree classifier

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
    # Boosting: n_estimators = 50, ccp_alpha = 0.005
    print("\nDataset: Rice | Algorithm: Boosting")
    print("=====================================")
    print("Hyperparameters")
    print("---------------")
    print("- n_estimators = 50")
    print("- ccp_alpha = 0.005")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="rice",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="boosting",
        n_estimators=50,
        ccp_alpha=0.005,
        random_state=1693854383,
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="rice", train_metrics=train_metrics)
    # Boosting: n_estimators = 100, ccp_alpha = 0.005
    print("\nDataset: Rice | Algorithm: Boosting")
    print("=====================================")
    print("Hyperparameters")
    print("---------------")
    print("- n_estimators = 100")
    print("- ccp_alpha = 0.005")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="rice",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="boosting",
        n_estimators=100,
        ccp_alpha=0.05,
        random_state=1693854383,
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="rice", train_metrics=train_metrics)
    # Boosting: n_estimators = 200, ccp_alpha = 0.005
    print("\nDataset: Rice | Algorithm: Boosting")
    print("=====================================")
    print("Hyperparameters")
    print("---------------")
    print("- n_estimators = 200")
    print("- ccp_alpha = 0.005")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="rice",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="boosting",
        n_estimators=200,
        ccp_alpha=0.005,
        random_state=1693854383,
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="rice", train_metrics=train_metrics)
    # Boosting: n_estimators = 200, ccp_alpha = 0.002
    print("\nDataset: Rice | Algorithm: Boosting")
    print("=====================================")
    print("Hyperparameters")
    print("---------------")
    print("- n_estimators = 200")
    print("- ccp_alpha = 0.002")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="rice",
        k=5,
        train_fracs=np.linspace(0.1, 1, 50),
        model_type="boosting",
        n_estimators=200,
        ccp_alpha=0.002,
        random_state=1693854383,
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="rice", train_metrics=train_metrics)

if execute_bank:
    # Bank Dataset
    # ------------
    # Boosting: n_estimators = 50, ccp_alpha = 0.005
    print("\nDataset: Bank | Algorithm: Boosting")
    print("=====================================")
    print("Hyperparameters")
    print("---------------")
    print("- n_estimators = 50")
    print("- ccp_alpha = 0.005")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="bank",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="boosting",
        n_estimators=50,
        ccp_alpha=0.005,
        random_state=1693854383,
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="bank", train_metrics=train_metrics)
    # Boosting: n_estimators = 100, ccp_alpha = 0.005
    print("\nDataset: Bank | Algorithm: Boosting")
    print("=====================================")
    print("Hyperparameters")
    print("---------------")
    print("- n_estimators = 100")
    print("- ccp_alpha = 0.005")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="bank",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="boosting",
        n_estimators=100,
        ccp_alpha=0.005,
        random_state=1693854383,
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="bank", train_metrics=train_metrics)
    # Boosting: n_estimators = 200, ccp_alpha = 0.005
    print("\nDataset: Bank | Algorithm: Boosting")
    print("=====================================")
    print("Hyperparameters")
    print("---------------")
    print("- n_estimators = 200")
    print("- ccp_alpha = 0.005")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="bank",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="boosting",
        n_estimators=200,
        ccp_alpha=0.005,
        random_state=1693854383,
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="bank", train_metrics=train_metrics)
    # Boosting: n_estimators = 200, ccp_alpha = 0.002
    print("\nDataset: Bank | Algorithm: Boosting")
    print("=====================================")
    print("Hyperparameters")
    print("---------------")
    print("- n_estimators = 200")
    print("- ccp_alpha = 0.002")
    print("Training")
    print("--------")
    train_metrics = Trainer()(
        data_name="bank",
        k=5,
        train_fracs=np.linspace(0.1, 1, 30),
        model_type="boosting",
        n_estimators=200,
        ccp_alpha=0.002,
        random_state=1693854383,
        )
    PlotTrainMetrices()(train_metrics=train_metrics)
    print("Testing")
    print("-------")
    Tester()(data_name="bank", train_metrics=train_metrics)
