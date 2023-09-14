# This module defines the ploting classes

import os

import matplotlib.pyplot as plt
import numpy as np


class PlotTrainMetrices:
    """Plot the metrices generated during training"""

    def __call__(self, train_metrics):
        """Object
        Parameters:
            train_metrics: Combined training metrices (Dict)
                {
                    "model_type_w_hparams": string
                    "train_sizes": np.ndarray[K-folds, train-fracs],
                    "train_losses": np.ndarray[K-folds, train-fracs],
                    "val_losses": np.ndarray[K-folds, train-fracs],
                    "train_times": np.ndarray[K-folds, train-fracs]
                }
        """
        # Set and create the directory for saving plots
        plots_dir = os.path.join("..", "plots", train_metrics["model_type_w_hparams"])
        if not os.path.isdir(plots_dir):
            os.mkdir(plots_dir)

        # Get the mean training size, training losses, validation losses, and training times
        mean_training_size = np.mean(train_metrics["train_sizes"], axis=0)
        mean_training_loss = np.mean(train_metrics["train_losses"], axis=0)
        mean_val_loss = np.mean(train_metrics["val_losses"], axis=0)
        mean_train_times = np.mean(train_metrics["train_times"], axis=0)

        # Get the standard deviation of training losses, validation losses, and training times
        std_training_loss = np.std(train_metrics["train_losses"], axis=0)
        std_val_loss = np.std(train_metrics["val_losses"], axis=0)
        std_train_times = np.std(train_metrics["train_times"], axis=0)

        # Plot the train and validation loss
        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot()
        ax.plot(mean_training_size, mean_training_loss, label="Training loss")
        ax.fill_between(
            x=mean_training_size,
            y1=mean_training_loss + std_training_loss,
            y2=mean_training_loss - std_training_loss,
            alpha=0.3
            )
        ax.plot(mean_training_size, mean_val_loss, label="Validation loss")
        ax.fill_between(
            x=mean_training_size,
            y1=mean_val_loss + std_val_loss,
            y2=mean_val_loss - std_val_loss,
            alpha=0.3
        )
        ax.legend()
        ax.set_xlabel("Training data size")
        ax.set_ylabel("Cross validation loss")
        ax.set_title("Training and validation curves")
        fig.savefig(os.path.join(plots_dir, "training_curves.png"))
        plt.close(fig)

        # Plot the training times
        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot()
        ax.plot(mean_training_size, mean_train_times, label="Training time")
        ax.fill_between(
            x=mean_training_size,
            y1=mean_train_times + std_train_times,
            y2=mean_train_times - std_train_times,
            alpha=0.3
        )
        ax.legend()
        ax.set_title("Training times")
        ax.set_xlabel("Training data size")
        ax.set_ylabel("Training times")
        fig.savefig(os.path.join(plots_dir, "training_times.png"))
        plt.close(fig)

        print(f"- Plots related to model training saved at: {plots_dir}")


if __name__ == "__main__":

    # Test code
    import os
    import pickle as pkl

    print("\nTesting ploting:\n================")
    # Load the train metrics
    source_filename = os.path.join("..", "models", "svm-rice-probability_True", "combined_train_metrics.pkl")
    with open(source_filename, "rb") as f:
        train_metrics = pkl.load(f)
    # Plots the train metrics
    PlotTrainMetrices()(train_metrics=train_metrics)
