# CS7641 Assignment 1: Supervised Learning

In this project, different machine learning algorithms are trained, validated, tested, and compared on two classification datasets


## Project source code

- The project's source code can be found [here](https://github.gatech.edu/skumar691/cs7641-assignment-1)
- Login credential for [github.gatech.edu](https://github.gatech.edu/) is required

Execute the following command to clone the source code:
```bash
# When prompted, please enter the username & password
# for github.gatech.edu
git clone https://github.gatech.edu/skumar691/cs7641-assignment-1.git
```

**_NOTE_**: If one is unable to access [github.gatech.edu](https://github.gatech.edu/), the same source code is also exported to my personal github [here](https://github.com/jkSwapnil/cs7641-assignment-1). No credentials are needed to clone this one.


## Datasets

The following two classification problems are used:
- [Rice (Cammeo and Osmancik)](https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik)
- [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)

These datasets are already included in the project source code. So, there is no need to download them again.


## Required packages

The following python packages are needed for this project
- numpy==1.24.4
- pandas==2.0.3
- scikit-learn==1.3.0
- imbalanced-learn==0.11.0
- matplotlib==3.7.3
- tqdm==4.66.1

After cloning the repo, these packages can be installed using the following command
```bash
pip install -r requirements.txt
```


## Execute training, validation, and testing

The models and related plots are saved in [./models](./models/) and [./plots](./plots/) directories respectively. One can choose to download the pre-trained models and plots from [here](https://www.dropbox.com/scl/fi/7ief8k2gnvrw4muux19sg/cs7641-assignment-1.zip?rlkey=5sg361gdzcifulc2cqzyshu6m&dl=0). These models can also be trained and tested and the plots can be created by executing the following commands.

### Decision Tree Classifier
The decision tree classifier can be trained and tested by executing one of the following commands
```bash
# Train and test on both the datasets
python main_decision_tree.py

# Train and test only on Rice dataset
python main_decision_tree.py rice

# Train and test only on Bank dataset
python main_decision_tree.py bank
```

### Boosting
The boosting models can be trained and tested by executing one of the following commands
```bash
# Train and test on both the datasets
python main_boosting.py

# Train and test only on Rice dataset
python main_boosting.py rice

# Train and test only on Bank dataset
python main_boosting.py bank
```

### k-Nearest Neighbors
The KNN models can be trained and tested by executing one of the following commands
```bash
# Train and test on both the datasets
python main_knn.py

# Train and test only on Rice dataset
python main_knn.py rice

# Train and test only on Bank dataset
python main_knn.py bank
```

### Support Vector Machines
The SVM models can be trained and tested by executing one of the following commands
```bash
# Train and test on both the datasets
python main_svm.py

# Train and test only on Rice dataset
python main_svm.py rice

# Train and test only on Bank dataset
python main_svm.py bank
```

### Neural Networks
The neural networks can be trained and tested by executing one of the following commands
```bash
# Train and test on both the datasets
python main_nn.py

# Train and test only on Rice dataset
python main_nn.py rice

# Train and test only on Bank dataset
python main_nn.py bank
```
