# This module defines interfaces and classes implementing the datasets

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder


class Data:
    """Dataset interface
    Attributes:
        - self.name: Name of the dataset (string)
        - self.size: Number of samples in the data: (int)
        - self.size_train: Number of samples in the train data: (int)
        - self.size_test: Number of samples in the test data: (int)
        - self.x_train: Input feature of the train data (numpy.array)
        - self.y_train: Labels of the train data (numpy.array)
        - self.x_test: Input feature of the test data (numpy.array)
        - self.y_test: Labels of the test data (numpy.array)
        - self.feature_names: Names of the features in 'x' (list)
        - self.label_names: Names of the labels in 'y' (list)

    Methods:
        __init__(): Load and pre-proecess the dataset
    """

    def __init__(self, name):
        """Constructor
        Parameters:
            name: name of the dataset (string)
        """
        self.name = name
        self.size = None
        self.size_train = None
        self.size_test = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.feature_names: None
        self.label_names = None

    def get_train(self):
        """Get the train part of the data
        Returns:
            [x_train, y_train]
            x_train: Input feature of train data (numpy.array)
            y_train: Label of the train data (numpy.array)
        """
        return [self.x_train, self.y_train]

    def get_test(self):
        """Get the test part of the data
        Returns:
            [x_test, y_test]
            x_test: Input feature of test data (numpy.array)
            y_test: Label of the test data (numpy.array)
        """
        return [self.x_test, self.y_test]


class RiceData(Data):
    """Rice (Cammeo and Osmancik)

    - This class implements loading and processing of 'Rice (Cammeo and Osmancik)' dataset.
    - URL: https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik
    - Output labels: { b"Cammeo": 0, b"Osmancik": 1 }
    - train-test-split: 80-20 split
    """

    def __init__(self, name="Rice (Cammeo and Osmancik)"):
        """Constructor
        Parameters:
            name: Name of the dataset (string)
        """
        super(RiceData, self).__init__(name=name)
        df = pd.DataFrame(arff.loadarff("../data/rice_cammeo_osmancik.arff")[0])
        self.size = len(df)
        x = df[df.columns.difference(["Class"])].values
        y = df["Class"].map(lambda x: '0' if(x == b'Cammeo') else '1').astype(int).values
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, random_state=1693854383, shuffle=True, stratify=y
            )
        self.size_train = len(self.x_train)
        self.size_test = len(self.x_test)
        self.feature_names = list(df.columns.difference(["Class"]))
        self.label_names = [b'Cammeo', b"Osmancik"]


class BankData(Data):
    """Bank Marketing

    - This class implements loading and processing splitting of 'Bank Marketing' dataset.
    - URL: https://archive.ics.uci.edu/dataset/222/bank+marketing
    - Output labels: { "no": 0, "yes": 1 }
    - train-test-split: 80-20 split
    """

    def __init__(self, name="Bank Marketing"):
        """Constructor
        Parameters:
            name: Name of the dataset (string)
        """
        super(BankData, self).__init__(name=name)
        df = pd.read_csv("../data/bank_marketing.csv", sep=';')

        # Set the size of the dataset
        self.size = len(df)

        # Perform one-hot-encoding
        # And generate the features
        one_hot_encoder = OneHotEncoder()
        # - 1. Encode "job" as One-Hot
        job_encoding = pd.DataFrame(
            one_hot_encoder.fit_transform(df[["job"]].values).todense(),
            columns=[
                "job_admin.",
                "job_blue-collar",
                "job_entrepreneur",
                "job_housemaid",
                "job_management",
                "job_retired",
                "job_self-employed",
                "job_services",
                "job_student",
                "job_technician",
                "job_unemployed",
                "job_unknown"
            ]
        )
        df = df.drop(["job"], axis=1)
        df = df.join(job_encoding, how="inner")
        # - 2. Encode "marital" as One-Hot
        marital_encoding = pd.DataFrame(
            one_hot_encoder.fit_transform(df[["marital"]].values).todense(),
            columns=["marital_divorced", "marital_married", "marital_single"]
        )
        df = df.drop(["marital"], axis=1)
        df = df.join(marital_encoding, how="inner")
        # - 3. Encode "education" as One-Hot
        education_encoding = pd.DataFrame(
            one_hot_encoder.fit_transform(df[["education"]].values).todense(),
            columns=["education_primary", "education_secondary", "education_tertiary", "education_unknown"]
        )
        df = df.drop(["education"], axis=1)
        df = df.join(education_encoding, how="inner")
        # - 4. Encode "default" as One-Hot
        default_encoding = pd.DataFrame(
            one_hot_encoder.fit_transform(df[["default"]].values).todense(),
            columns=["default_no", "default_yes"]
        )
        df = df.drop(["default"], axis=1)
        df = df.join(default_encoding, how="inner")
        # - 5. Encode "housing" as One-Hot
        housing_encoding = pd.DataFrame(
            one_hot_encoder.fit_transform(df[["housing"]].values).todense(),
            columns=["housing_no", "housing_yes"]
        )
        df = df.drop(["housing"], axis=1)
        df = df.join(housing_encoding, how="inner")
        # - 6. Encode "loan" as One-Hot
        loan_encoding = pd.DataFrame(
            one_hot_encoder.fit_transform(df[["loan"]].values).todense(),
            columns=["loan_no", "loan_yes"]
        )
        df = df.drop(["loan"], axis=1)
        df = df.join(loan_encoding, how="inner")
        # - 7. Encode "contact" as One-Hot
        contact_encoding = pd.DataFrame(
            one_hot_encoder.fit_transform(df[["contact"]].values).todense(),
            columns=["contact_cellular", "contact_telephone", "contact_unknown"]
        )
        df = df.drop(["contact"], axis=1)
        df = df.join(contact_encoding, how="inner")
        # - 8. Encode "month" as Ordinal
        label_encoding_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        df["month"] = df["month"].map(lambda x: label_encoding_map[x])
        # - 9. Encode "poutcome" as One-Hot
        poutcome_encoding = pd.DataFrame(
            one_hot_encoder.fit_transform(df[["poutcome"]].values).todense(),
            columns=["poutcome_failure", "poutcome_other", "poutcome_success", "poutcome_unknown"]
        )
        df = df.drop(["poutcome"], axis=1)
        df = df.join(poutcome_encoding, how="inner")

        # Get features and labels
        # Set the output labels as: { "no": 0, "yes": 1 }
        # Perform the train-test split
        x = df[df.columns.difference(["y"])].astype("float64").values
        y = df["y"].map({"no": 0, "yes": 1}).values
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, random_state=1693854383, shuffle=True, stratify=y
            )
        self.size_train = len(self.x_train)
        self.size_test = len(self.x_test)

        # Get the feature and label names
        self.feature_names = list(df.columns.difference(["y"]))
        self.label_names = ["no", "yes"]


class DataStats:
    """Print basic statistics on the dataset.

    Statistics such as name, size, class imbalance, no. of features, feature names, and label names of the dataset.
    """

    def __call__(self, data):
        """Print statistics of the datasets.
        Parameters:
            data: Dataset object for which to print the statistics (Data)
        """
        print(f"\n{data.name}:\n--------------------------")
        print(f"- Dataset size: {data.size}")
        print(f"- Train dataset size: {data.size_train}")
        for idx, _ in enumerate(data.label_names):
            print(f"\t- {_} label count: {np.sum(data.y_train == idx)}")
        print(f"- Test dataset size: {data.size_test}")
        for idx, _ in enumerate(data.label_names):
            print(f"\t- {_} label count: {np.sum(data.y_test == idx)}")
        print(f"- Number of features: {len(data.feature_names)}")
        print(f"- Feature names: {data.feature_names}")
        print(f"- Label names: {data.label_names}\n")


class DataSplit:
    """Stratified K-Fold splitting on the train dataset"""

    def __init__(self, k):
        """Constructor
        Parameters:
            k: Number of splits in Stratified K-Fold (int)
        """
        self.k = k

    def __call__(self, data):
        """Generate splits for cross validation on train dataset
        Parameters:
            data: Dataset object for which to print the statistics (Data)
        Yeilds:
            [x_train, x_val, y_train, y_val]: Each element is np.ndarray
        """
        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=1693854383)
        for (train_index, val_index) in skf.split(data.x_train, data.y_train):
            yield [
                data.x_train[train_index],
                data.x_train[val_index],
                data.y_train[train_index],
                data.y_train[val_index]
            ]


if __name__ == "__main__":

    # Testing code
    print("\nTesting the dataset implementation:\n====================================")

    # Rice (Cammeo and Osmancik) dataset
    data = RiceData()
    DataStats()(data)
    for x_train, x_test, y_train, y_test in DataSplit(k=5)(data=data):
        pass

    # Bank Marketing dataset
    data = BankData()
    DataStats()(data)
    for x_train, x_test, y_train, y_test in DataSplit(k=5)(data=data):
        pass
