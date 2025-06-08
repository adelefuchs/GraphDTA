import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def normalize_z_score(df, column, mean, std):
    """Apply Z-score normalization to a given column."""
    df[column] = (df[column] - mean) / std
    return df


def load_data(train_csv, test_csv, pred_csv):
    """Load and merge true affinities with predicted scores, then normalize."""
    train_df = pd.read_csv(train_csv, header=0)
    test_df = pd.read_csv(test_csv, header=0)
    pred_df = pd.read_csv(pred_csv, names=["True_Label", "Predicted_Value"], header=0)

    return train_df, test_df, pred_df


def compute_quartile_deviation(pred_df, true_test_df):
    """Compute mean signed difference from test quartile means."""
    true_test_df = true_test_df.copy()
    pred_df = pred_df.copy()

    true_test_df["Quartile"] = pd.qcut(
        true_test_df["affinity"], q=4, labels=False, duplicates="drop"
    )
    quartile_means = true_test_df.groupby("Quartile")["affinity"].mean()

    pred_df["Quartile"] = pd.qcut(
        pred_df["Predicted_Value"], q=4, labels=False, duplicates="drop"
    )
    pred_df["Quartile_Mean"] = pred_df["Quartile"].map(quartile_means)

    return (pred_df["Predicted_Value"] - pred_df["Quartile_Mean"]).mean()



def compute_distribution_shifts(davis_train_df, pharos_train_df, davis_test_df, pharos_test_df):
    """Compute Wasserstein distances between relevant distributions."""

    # Train to test shifts
    shift_davis_train_test = wasserstein_distance(
        davis_train_df["affinity"], davis_test_df["affinity"]
    )
    shift_pharos_train_test = wasserstein_distance(
        pharos_train_df["affinity"], pharos_test_df["affinity"]
    )

    # Cross-shifts between training and testing
    shift_davis_train_pharos_test = wasserstein_distance(
        davis_train_df["affinity"], pharos_test_df["affinity"]
    )
    shift_pharos_train_davis_test = wasserstein_distance(
        pharos_train_df["affinity"], davis_test_df["affinity"]
    )

    # Train-to-train shift
    shift_train_to_train = wasserstein_distance(
        davis_train_df["affinity"], pharos_train_df["affinity"]
    )

    # Test-to-test shift (as before)
    shift_test_to_test = wasserstein_distance(
        davis_test_df["affinity"], pharos_test_df["affinity"]
    )

    return {
        "train_davis→test_davis": shift_davis_train_test,
        "train_pharos→test_pharos": shift_pharos_train_test,
        "train_davis→test_pharos": shift_davis_train_pharos_test,
        "train_pharos→test_davis": shift_pharos_train_davis_test,
        "train_davis↔train_pharos": shift_train_to_train,
        "test_davis↔test_pharos": shift_test_to_test,
    }



def compute_csv_metric(
    train_df, davis_test_df, davis_pred_df, pharos_train_df, pharos_test_df, pharos_pred_df
):
    """Compute Cold Start Vulnerability metric considering all shifts."""
    delta_q_davis = compute_quartile_deviation(davis_pred_df, davis_test_df)
    print("Quartile Deviation between true and predicted Davis: ", delta_q_davis)
    delta_q_pharos = compute_quartile_deviation(pharos_pred_df, pharos_test_df)
    print("Quartile Deviation between true and predicted Pharos: ", delta_q_pharos)
    
    shifts = compute_distribution_shifts(
        train_df, pharos_train_df, davis_test_df, pharos_test_df
    )

    shift_train_davis = shifts["train_davis→test_davis"]
    shift_train_pharos = shifts["train_pharos→test_pharos"]
    shift_davis_pharos = shifts["test_davis↔test_pharos"]

    delta_q_diff = abs(delta_q_davis - delta_q_pharos)

    dist_shift_total = shift_train_pharos + shift_train_davis + shift_davis_pharos
    print("Total distribution shift (just sums): ", dist_shift_total)
    csv = (
        delta_q_diff / dist_shift_total if dist_shift_total > 0 else np.nan
    )  # Avoid division by zero
    print("CSVM = ", delta_q_davis, " - ", delta_q_pharos , " / ", dist_shift_total)
    return csv, shift_train_davis, shift_train_pharos, shift_davis_pharos


def visualize_quartiles(
    train_df, davis_test_df, davis_pred_df, pharos_test_df, pharos_pred_df
):
    """Visualize and print quartile statistics for training, DAVIS, and Pharos datasets."""
    print("Training Data Summary:")
    print(train_df["affinity"].describe())

    print("\nDAVIS Test Data Summary:")
    print(davis_test_df["affinity"].describe())

    print("\nDAVIS Predictions Summary:")
    print(davis_pred_df["True_Label"].describe())

    print("\nPharos Test Data Summary:")
    print(pharos_test_df["affinity"].describe())

    print("\nPharos Predictions Summary:")
    print(pharos_pred_df["Predicted_Value"].describe())

    fig, axes = plt.subplots(1, 5, figsize=(30, 6))

    datasets = [
        ("Training Data", train_df, "affinity"),
        ("DAVIS Test Data", davis_test_df, "affinity"),
        ("DAVIS Predictions", davis_pred_df, "Predicted_Value"),
        ("Pharos Test Data", pharos_test_df, "affinity"),
        ("Pharos Predictions", pharos_pred_df, "Predicted_Value"),
    ]

    for i, (title, df, col) in enumerate(datasets):
        df["Quartile"] = pd.qcut(df[col], q=4, labels=False, duplicates="drop")

        # Plot quartiles
        axes[i].boxplot(
            [df[df["Quartile"] == q][col] for q in range(df["Quartile"].nunique())]
        )
        axes[i].set_title(title)
        axes[i].set_xlabel("Quartile")
        axes[i].set_ylabel(col)

    plt.tight_layout()
    plt.show()


def main(train_csv, davis_test_csv, davis_pred_csv, pharos_test_csv, pharos_pred_csv, pharos_train_csv=None):
    
    train_df, davis_test_df, davis_pred_df = load_data(
        train_csv, davis_test_csv, davis_pred_csv
    )
    
    # Use separate train set for Pharos if provided, else use the same as training data
    if pharos_train_csv:
        pharos_train_df = pd.read_csv(pharos_train_csv, header=0)
    else:
        pharos_train_df = train_df.copy()

    # Load Pharos test and prediction data
    pharos_test_df, pharos_pred_df = load_data(
        None, pharos_test_csv, pharos_pred_csv
    )

    # # # Compute mean and std from the training set for normalization
    # train_mean, train_std = train_df["affinity"].mean(), train_df["affinity"].std()
    
    # #NOT Sure IF I SHOULD NOrMALIZE HERE
    # train_df["affinity"] = (train_df["affinity"] - train_mean) / train_std
    # davis_test_df["affinity"] = (davis_test_df["affinity"] - train_mean) / train_std
    # pharos_test_df["affinity"] = (pharos_test_df["affinity"] - train_mean) / train_std
    # davis_pred_df["Predicted_Value"] = (davis_pred_df["Predicted_Value"] - train_mean) / train_std
    # pharos_pred_df["Predicted_Value"] = (pharos_pred_df["Predicted_Value"] - train_mean) / train_std


    csv, shift_train_davis, shift_train_pharos, shift_davis_pharos = compute_csv_metric(
        train_df, davis_test_df, davis_pred_df,
        pharos_train_df, pharos_test_df, pharos_pred_df
    )

    print(f"CSV Metric: {csv}")
    print(f"Train → DAVIS Shift: {shift_train_davis}")
    print(f"Train → Pharos Shift: {shift_train_pharos}")
    print(f"DAVIS → Pharos Shift: {shift_davis_pharos}")

    visualize_quartiles(
        train_df, davis_test_df, davis_pred_df, pharos_test_df, pharos_pred_df
    )


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("train_csv", type=str, help="Path to training CSV file")
#     parser.add_argument("davis_test_csv", type=str, help="Path to DAVIS test CSV file")
#     parser.add_argument(
#         "davis_pred_csv", type=str, help="Path to DAVIS predictions CSV file"
#     )
#     parser.add_argument(
#         "pharos_test_csv", type=str, help="Path to Pharos test CSV file"
#     )
#     parser.add_argument(
#         "pharos_pred_csv", type=str, help="Path to Pharos predictions CSV file"
#     )

#     args = parser.parse_args()
#     main(
#         args.train_csv,
#         args.davis_test_csv,
#         args.davis_pred_csv,
#         args.pharos_test_csv,
#         args.pharos_pred_csv,
#     )
