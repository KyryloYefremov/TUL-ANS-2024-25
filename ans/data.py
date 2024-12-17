import math
from typing import Callable, Iterator, Optional

import numpy as np
import torch


class BatchLoader:

    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            batch_size: Optional[int] = None,
            shuffle: bool = False,
            device: str = 'cpu'
    ) -> None:
        """
        Args:
            dataset: indexable torch Dataset returning either a tuple (input,) or (input, target)
            batch_size: How many samples in batch
            shuffle: If True, then the data should be randomly reordered on each __iter__
            device: torch device name, e.g. 'cpu' or 'cuda:0', 'cuda:1', etc.
        """
        self.dataset = dataset
        self.batch_size = batch_size or len(dataset)
        self.shuffle = shuffle
        self.device = device

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        """
        Returns:
            batch: If unsupervised (i.e. self.y is None), return single element tuple (x[batch_ids],). If supervised
                   (i.e. self.y is torch.Tensor), return the pair (x[batch_ids], y[batch_ids])
        """
        
        dataset_size = len(self.dataset)
        indeces = torch.arange(dataset_size)
        if self.shuffle:
            indeces = indeces[torch.randperm(dataset_size)]

        for start_idx in range(0, dataset_size, self.batch_size):
            batch_indeces = indeces[start_idx : start_idx + self.batch_size]
            batch = [self.dataset[i] for i in batch_indeces]

            if len(batch[0]) == 2:
                x, y = zip(*batch)
                x = torch.stack(x)
                y = torch.tensor(y, dtype=torch.int64)
                # change device for tensors
                if self.device != 'cpu':
                    x = x.to(device=self.device)
                    y = y.to(device=self.device)
                yield x, y
            else:
                x, = zip(*batch)
                x = torch.stack(x)
                # change device for tensors
                if self.device != 'cpu':
                    x = x.to(device=self.device)    
                yield (x,)
            

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}:\n" \
               f"    num_batches: {len(self)}\n" \
               f"    batch_size: {self.batch_size}\n"