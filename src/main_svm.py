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
