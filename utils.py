import pandas as pd
import torch
import torch.nn as nn
import math


def load_data_to_tensor(csv_path, target_column, test_size=0.2, random_seed=42):
    """
    Reads a CSV file, splits it into train/test sets, and converts the data to PyTorch tensors.

    Parameters:
    - csv_path (str): Path to the CSV file.
    - target_column (str): Name of the target column in the CSV file.
    - test_size (float): Fraction of data to be used as the test set (default is 0.2, meaning 20% test, 80% train).
    - random_seed (int): Seed for random shuffling (default is 42).

    Returns:
    - X_train_tensor (Tensor): Training features as a PyTorch tensor.
    - X_test_tensor (Tensor): Testing features as a PyTorch tensor.
    - y_train_tensor (Tensor): Training labels as a PyTorch tensor.
    - y_test_tensor (Tensor): Testing labels as a PyTorch tensor.
    """

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Separate features and labels
    X = df.drop(columns=target_column)
    y = df[target_column]

    # Shuffle the data
    shuffled_df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Re-split the shuffled data
    X_shuffled = shuffled_df.drop(columns=target_column)
    y_shuffled = shuffled_df[target_column]

    # Calculate the index to split data
    train_size = int((1 - test_size) * len(shuffled_df))

    # Split the data into train and test
    X_train = X_shuffled[:train_size]
    X_test = X_shuffled[train_size:]
    y_train = y_shuffled[:train_size]
    y_test = y_shuffled[train_size:]

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1,1)  # Change dtype if necessary
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1,1)  # Change dtype if necessary

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

def initize_xavier(m):
    if isinstance(m, nn.Linear):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
        std = math.sqrt(2.0 / (fan_in + fan_out))

        # Xavier normal for weights
        nn.init.normal_(m.weight, mean=0.0, std=std)

        # Bias from same distribution (same covariance)
        if m.bias is not None:
            nn.init.normal_(m.bias, mean=0.0, std=std)