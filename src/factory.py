# This module defines the factory functions\

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from datasets import RiceData, BankData
from models import Model

def get_data(dtype):
    """Dataset factory method
    Parameters:
        dtype: "rice" or "bank" (string)
    Returns:
        Data object
    """
    if (dtype == "rice"):
        return RiceData()
    elif (dtype == "bank"):
        return BankData()
    else:
        raise Exception(f"Dataset '{dtype}' not supported !")

def get_model(alg_name, **hparams):
    """Model factory
    Parameters:
        alg_name: Name of the algorithm (string)
        hparams: Hyper-parameters (dict) (Optional)
    Returns:
        Model object
    """
    if alg_name == "decision_tree":
        return Model(DecisionTreeClassifier(**hparams))
    elif alg_name == "neural_network":
        return Model(MLPClassifier(**hparams))
    elif alg_name == "boosting":
        return Model(GradientBoostingClassifier(**hparams))
    elif alg_name == "svm":
        return Model(SVC(**hparams))
    elif alg_name == "knn":
        return Model(KNeighborsClassifier(**hparams))
    else:
        raise Exception(f"Algorithm '{alg_name}' is not supported !")


if __name__ == "__main__":

    # Test code
    from datasets import DataStats, DataSplit

    # Test dataset factory
    print("\nTesting dataset factory:\n============================")
    # Rice (Cammeo and Osmancik) dataset
    data = get_data(dtype="rice")
    DataStats()(data)
    for x_train, x_test, y_train, y_test in DataSplit(k=5)(data=data):
        pass
    # Bank Marketing dataset
    data = get_data(dtype="bank")
    DataStats()(data)
    for x_train, x_test, y_train, y_test in DataSplit(k=5)(data=data):
        pass

    # Test model factory
    print("\n\nTesting model factory:\n======================")
    model = get_model(alg_name="decision_tree", ccp_alpha=0.2)
    model = get_model(alg_name="neural_network")
    model = get_model(alg_name="boosting")
    model = get_model(alg_name="svm")
    model = get_model(alg_name="knn")
    print("- done\n")
