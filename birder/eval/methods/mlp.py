"""
MLP probe for multi-label classification evaluation

Trains a 2-layer MLP on pre-extracted embeddings to predict binary traits.
Architecture follows the AwA2 (Animals with Attributes 2) design:
Linear -> Dropout -> Linear (no activation between layers).
"""

import logging

import numpy as np
import numpy.typing as npt
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


class MLPProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.5) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# pylint: disable=too-many-locals
def train_mlp(
    train_features: npt.NDArray[np.float32],
    train_labels: npt.NDArray[np.float32],
    num_classes: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float = 1e-4,
    step_size: int = 20,
    hidden_dim: int = 512,
    dropout: float = 0.5,
    seed: int = 0,
) -> MLPProbe:
    """
    Train MLP probe on embeddings for multi-label classification

    Parameters
    ----------
    train_features
        Training features of shape (n_train, embedding_dim).
    train_labels
        Training labels of shape (n_train, num_classes), float values 0/1.
    num_classes
        Number of output classes (traits).
    device
        Device to train on.
    epochs
        Number of training epochs.
    batch_size
        Batch size for training.
    lr
        Learning rate.
    step_size
        Number of epochs between learning rate decay steps.
    hidden_dim
        Hidden layer dimension.
    dropout
        Dropout probability.
    seed
        Random seed for reproducibility.

    Returns
    -------
    Trained MLPProbe model.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = train_features.shape[1]
    model = MLPProbe(input_dim, num_classes, hidden_dim, dropout).to(device)

    x_train = torch.from_numpy(train_features).float()
    y_train = torch.from_numpy(train_labels).float()

    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

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

    return model  # type: ignore[no-any-return]


def evaluate_mlp(
    model: MLPProbe,
    test_features: npt.NDArray[np.float32],
    test_labels: npt.NDArray[np.float32],
    batch_size: int = 128,
    device: torch.device = torch.device("cpu"),
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], float]:
    """
    Evaluate MLP probe on test set

    Parameters
    ----------
    model
        Trained MLPProbe model.
    test_features
        Test features of shape (n_test, embedding_dim).
    test_labels
        Test labels of shape (n_test, num_classes), float values 0/1.
    batch_size
        Batch size for inference.
    device
        Device to run on.

    Returns
    -------
    y_pred
        Predicted labels of shape (n_test, num_classes), int values 0/1.
    y_true
        True labels of shape (n_test, num_classes), int values 0/1.
    macro_f1
        Macro-averaged F1 score across all classes.
    """

    x_test = torch.from_numpy(test_features).float()
    y_test = torch.from_numpy(test_labels).float()

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
            preds = (torch.sigmoid(logits) > 0.5).int()
            all_preds.append(preds.cpu())
            all_labels.append(batch_y.int())

    y_pred = torch.concat(all_preds, dim=0).numpy()
    y_true = torch.concat(all_labels, dim=0).numpy()
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0.0)

    return (y_pred, y_true, float(macro_f1))
