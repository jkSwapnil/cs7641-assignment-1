# Main entrypoint for the neural network classifier

import numpy as np

from train import Trainer
from test import Tester
from plots import PlotTrainMetrices

# Rice Dataset
# ------------
# Neural network: learning_rate_init = 0.001, hidden_layer_sizes = [100, 100]
print("\nDataset: Rice | Algorithm: Neural network")
print("===========================================")
print("Hyperparameters")
print("---------------")
print("- learning_rate_init = 0.001")
print("- hidden_layer_sizes = [100, 100]")
print("Training")
print("--------")
train_metrics = Trainer()(
    data_name="rice",
    k=5,
    train_fracs=np.linspace(0.1, 1, 30),
    model_type="neural_network",
    learning_rate_init=0.001,
    hidden_layer_sizes=[100, 100, 100, 100],
    random_state=1693854383,
    )
PlotTrainMetrices()(train_metrics=train_metrics)
print("Testing")
print("-------")
Tester()(data_name="rice", train_metrics=train_metrics)



