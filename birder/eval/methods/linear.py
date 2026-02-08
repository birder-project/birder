"""
Linear probing on frozen embeddings

Trains a single linear layer with cross-entropy loss for classification evaluation.
"""

import logging

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


def train_linear_probe(
    train_features: npt.NDArray[np.float32],
    train_labels: npt.NDArray[np.int_],
    num_classes: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float = 1e-3,
    seed: int = 0,
) -> nn.Linear:
    """
    Train a linear probe on frozen embeddings

    Parameters
    ----------
    train_features
        Training features of shape (n_train, embedding_dim).
    train_labels
        Training labels of shape (n_train,), integer class indices.
    num_classes
        Number of output classes.
    device
        Device to train on.
    epochs
        Number of training epochs.
    batch_size
        Batch size for training.
    lr
        Learning rate.
    seed
        Random seed for reproducibility.

    Returns
    -------
    Trained linear layer.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = train_features.shape[1]
    model = nn.Linear(input_dim, num_classes).to(device)

    x_train = torch.from_numpy(train_features).float()
    y_train = torch.from_numpy(train_labels).long()

    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataset)
            logger.debug(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    return model


def evaluate_linear_probe(
    model: nn.Linear,
    test_features: npt.NDArray[np.float32],
    test_labels: npt.NDArray[np.int_],
    batch_size: int = 128,
    device: torch.device = torch.device("cpu"),
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """
    Evaluate linear probe on test set

    Parameters
    ----------
    model
        Trained linear layer.
    test_features
        Test features of shape (n_test, embedding_dim).
    test_labels
        Test labels of shape (n_test,), integer class indices.
    batch_size
        Batch size for inference.
    device
        Device to run on.

    Returns
    -------
    y_pred
        Predicted labels for test samples.
    y_true
        True labels for test samples (same as test_labels).
    """

    x_test = torch.from_numpy(test_features).float()
    y_test = torch.from_numpy(test_labels).long()

    dataset = TensorDataset(x_test, y_test)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    model.eval()
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.inference_mode():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(batch_y)

    y_pred = torch.concat(all_preds, dim=0).numpy().astype(np.int_)
    y_true = torch.concat(all_labels, dim=0).numpy().astype(np.int_)

    return (y_pred, y_true)
