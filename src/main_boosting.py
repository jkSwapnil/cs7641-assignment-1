# Main entrypoint

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from train import Trainer
from test import Tester
from plots import PlotTrainMetrices

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
    k=1,
    train_fracs=np.linspace(0.1, 1, 30),
    model_type="boosting",
    n_estimators = 50,
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
    k=1,
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
# Boosting: n_estimators = 200, ccp_alpha = 0
print("\nDataset: Rice | Algorithm: Boosting")
print("=====================================")
print("Hyperparameters")
print("---------------")
print("- n_estimators = 200")
print("- ccp_alpha = 0")
print("Training")
print("--------")
train_metrics = Trainer()(
    data_name="rice",
    k=1,
    train_fracs=np.linspace(0.1, 1, 50),
    model_type="boosting",
    n_estimators=200,
    ccp_alpha=0,
    random_state=1693854383,
    )
PlotTrainMetrices()(train_metrics=train_metrics)
print("Testing")
print("-------")
Tester()(data_name="rice", train_metrics=train_metrics)

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
    k=1,
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
    k=1,
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
    k=1,
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

# Boosting: n_estimators = 200, ccp_alpha = 0
print("\nDataset: Bank | Algorithm: Boosting")
print("=====================================")
print("Hyperparameters")
print("---------------")
print("- n_estimators = 200")
print("- ccp_alpha = 0")
print("Training")
print("--------")
train_metrics = Trainer()(
    data_name="bank",
    k=1,
    train_fracs=np.linspace(0.1, 1, 30),
    model_type="boosting",
    n_estimators=200,
    ccp_alpha=0,
    random_state=1693854383,
    )
PlotTrainMetrices()(train_metrics=train_metrics)
print("Testing")
print("-------")
Tester()(data_name="bank", train_metrics=train_metrics)
