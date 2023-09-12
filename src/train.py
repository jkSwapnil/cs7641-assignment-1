# Modules implementing training class

import os
import pickle as pkl
import time

import numpy as np
from tqdm import tqdm

from datasets import DataSplit
from factory import get_model
from sklearn.metrics import log_loss


class Trainer:
    """Trainer class

    This class implements training and validation of the model.
    Training data is progressively used to create the train-validation-curve
    Stratified K-Fold cross validation is used
    """

    def __call__(self, data, k, train_fracs, model_type, **hparams):
        """Object call to implement training.
        Paramters:
            data: Data to train on (Data)
            k: Number of splits in k-fold (int)
            train_fracs: Utilization of the training data (np.ndarray)
            model_type Type of model to train (string)
            **hparams: Hyperparameters of the model (dict) (Optional)
        Returns:
            Combined training metrices (Dict)
                {
                    "train_sizes": np.ndarray[K-folds, train-fracs],
                    "train_losses": np.ndarray[K-folds, train-fracs],
                    "val_losses": np.ndarray[K-folds, train-fracs],
                    "train_times": np.ndarray[K-folds, train-fracs]
                }
        """
        # Define and create model paths
        model_dir = os.path.join("..", "models", model_type)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        # All the combined metrics listed below is saved as np.ndarray
        # Shape = [No. of folds, No. of train-fracs]
        train_sizes = []  # List of train sizes in all cross-validations
        train_losses = []  # List of train losses in all cross-validations
        val_losses = []  # List of validation losses in all cross-validations
        train_times = []  # List of train times in all cross-validations
        # Run training and validation on each cross-validation split
        print(f"- Number of folds in cross validation: {k}")
        for kid, (x_train, x_val, y_train, y_val) in enumerate(DataSplit(k=5)(data=data)):
            print(f"- Fold: {kid}")
            cv_train_size = []  # List of train size for this cross-validation split
            cv_train_loss = []  # List of train loss for this cross-validation split
            cv_val_loss = []  # List of validation losss for this cross-validation split
            cv_train_time = []  # List of training time for this cross-validation split
            # Loop over the training size fractions
            for tfrac in tqdm(train_fracs):
                # Train size for the current fraction
                tfrac_size = int(tfrac * len(x_train))
                cv_train_size.append(tfrac_size)
                # Get the training data according to the current train fraction
                x_train_frac = x_train[0:tfrac_size]
                y_train_frac = y_train[0:tfrac_size]
                # Create and train the model
                model = get_model(alg_name=model_type, **hparams)
                train_start_time = time.time()
                model.fit(x_train_frac, y_train_frac)
                train_end_time = time.time()
                cv_train_time.append(train_end_time - train_start_time)
                # Evaluate the train and validation loss
                y_train_frac_pred_proba = model.predict_proba(x_train_frac)
                tloss = log_loss(y_true=y_train_frac, y_pred=y_train_frac_pred_proba)
                cv_train_loss.append(tloss)
                y_val_pred_proba = model.predict_proba(x_val)
                vloss = log_loss(y_true=y_val, y_pred=y_val_pred_proba)
                cv_val_loss.append(vloss)
                # Save the trained model
                model_file = os.path.join(model_dir, f"kfold_{kid}_tfrac_size_{tfrac_size}.pkl")
                with open(model_file, "wb") as f:
                    pkl.dump(model, f)
            # Add the fold's metrices to combined list
            train_sizes.append(cv_train_size)
            train_losses.append(cv_train_loss)
            val_losses.append(cv_val_loss)
            train_times.append(cv_train_time)
        # Convert the combined metrices to numpy array and save
        combined_train_metrics = {}
        combined_train_metrics["train_sizes"] = np.array(train_sizes)
        combined_train_metrics["train_losses"] = np.array(train_losses)
        combined_train_metrics["val_losses"] = np.array(val_losses)
        combined_train_metrics["train_times"] = np.array(train_times)
        metrics_path = os.path.join(model_dir, "combined_train_metrics.pkl")
        with open(metrics_path, "wb") as f:
            pkl.dump(combined_train_metrics, f)

        return combined_train_metrics


if __name__ == "__main__":

    # Test code
    from factory import get_data, get_model

    print("\nTesting trainer:\n================")
    data = get_data(dtype="rice")  # Get data from the factory
    trainer = Trainer()  # Trainer object
    out = trainer(data=data, k=5, train_fracs=np.linspace(0.1, 1, 90), model_type="svm", probability=True)
    # print(out)
    print("- done\n")
