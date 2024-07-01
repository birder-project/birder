from typing import Optional
from typing import Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F


def predict(
    net: Union[torch.nn.Module, torch.ScriptModule], inputs: torch.Tensor, return_embedding: bool = False
) -> tuple[npt.NDArray[np.float32], Optional[npt.NDArray[np.float32]]]:
    if return_embedding is True:
        embedding_tensor: torch.Tensor = net.embedding(inputs)
        out: npt.NDArray[np.float32] = F.softmax(net.classify(embedding_tensor), dim=1).cpu().numpy()
        embedding: Optional[npt.NDArray[np.float32]] = embedding_tensor.cpu().numpy()

    else:
        embedding = None
        out = F.softmax(net(inputs), dim=1).cpu().numpy()

    return (out, embedding)
