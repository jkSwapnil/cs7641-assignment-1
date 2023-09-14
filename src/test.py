# This module implments the model testing class

import json
import os
import pickle as pkl

from factory import get_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


class Tester:
    """Implements testing of the model"""

    def __call__(self, data_name, train_metrics):
        """Object call

        Test the best model on the test part of the dataset.
        The best model is determined as the model with the lowest validation loss.

        Parameters:
            data_name: Name of the dataset "rice" or "bank" (string)
            train_metrics: Combined training metrices (Dict)
                {
                    "model_type_w_hparams": string
                    "train_sizes": np.ndarray[K-folds, train-fracs],
                    "train_losses": np.ndarray[K-folds, train-fracs],
                    "val_losses": np.ndarray[K-folds, train-fracs],
                    "train_times": np.ndarray[K-folds, train-fracs]
                }
        """
        # Set the model base directory
        model_base_dir = os.path.join("..", "models", train_metrics["model_type_w_hparams"])

        # Get the best model's kfold and tfrac_size value
        # Point to the best model's pickle file
        argmin_row_idx = np.argmin(np.min(train_metrics["val_losses"], axis=1))
        argmin_col_idx = np.argmin(np.min(train_metrics["val_losses"], axis=0))
        kfold_value = str(argmin_row_idx)
        tfrac_size_value = train_metrics["train_sizes"][argmin_row_idx, argmin_col_idx]
        model_pkl_file = os.path.join(model_base_dir, f"kfold_{kfold_value}_tfrac_size_{tfrac_size_value}.pkl")

        # Load the best model
        with open(model_pkl_file, "rb") as f:
            model = pkl.load(f)

        # Get the test data and generate predictions
        data = get_data(data_name)
        x_test, y_test = data.get_test()
        y_pred = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)

        #  Evaluate the following metrics:
        # - Accuracy
        # - Precision, Recall, and F1-Score
        # - AUC-ROC
        test_results = {}
        percentage_accuracy = accuracy_score(y_test, y_pred)*100
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        print(f"- Accuracy: {percentage_accuracy:.2f}%")
        print(f"- Precision: {precision:.2f}")
        print(f"- Recall: {recall:.2f}")
        print(f"- F1 score: {f1:.2f}")
        print(f"- ROC AUC: {roc_auc:.2f}")
        # Saved the evaluted metrics
        test_results["accuracy"] = percentage_accuracy
        test_results["precision"] = precision
        test_results["recall"] = recall
        test_results["f1_score"] = f1
        test_results["roc_auc"] = roc_auc
        test_results_filepath = os.path.join(model_base_dir, "test_results.json")
        with open(test_results_filepath, "w") as f:
            json.dump(test_results_filepath, f)
        print(f"- Test results saved at: {test_results_filepath}")

        plots_base_dir =  os.path.join("..", "plots", train_metrics["model_type_w_hparams"])
        if not os.path.isdir(plots_base_dir):
            os.mkdir(plots_base_dir)
        # Plot the ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:,1])
        fig = plt.figure(figsize=(6.4,4.8))
        ax = fig.add_subplot(1,1,1)
        ax.plot(fpr, tpr, label="Model ROC curve")
        ax.plot([0, 0.5, 1], [0, 0.5, 1], "--", label="No skill ROC curve")
        ax.legend()
        ax.set_title("ROC curve")
        ax.set_ylabel("True positive rate")
        ax.set_xlabel("False positive rate")
        roc_curve_filepath = os.path.join(plots_base_dir, "roc_curve.png")
        fig.savefig(roc_curve_filepath)
        plt.close(fig)
        print(f"- ROC curve plot saved at: {roc_curve_filepath}")
        # Plot the confusion matrix
        cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
        cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.label_names)
        cm_disp.plot()
        fig = cm_disp.figure_
        fig.set_figwidth(8)
        fig.set_figheight(6)
        ax = cm_disp.ax_
        ax.set_title("Confusion matrix")
        confusion_matrix_filepath = os.path.join(plots_base_dir, "confusion_matrix.png")
        fig.savefig(confusion_matrix_filepath)
        plt.close(fig)
        print(f"- Confusion matrix plot saved at: {confusion_matrix_filepath}")


if __name__ == "__main__":

    # Test code
    import os
    import pickle as pkl

    print("\nTesting tester:\n================")
    # Load the train metrics
    source_filename = os.path.join("..", "models", "svm-rice-probability_True", "combined_train_metrics.pkl")
    with open(source_filename, "rb") as f:
        train_metrics = pkl.load(f)
    Tester()(data_name="rice", train_metrics=train_metrics)
