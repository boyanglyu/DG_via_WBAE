# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modified for DDP training
"""

import torch
from torch.utils.data.distributed import DistributedSampler

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, num_workers, rank, world_size, shuffle=True):
        super().__init__()

        # Use DistributedSampler

        self.sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
        batch_sampler = torch.utils.data.BatchSampler(
            self.sampler,
            batch_size=batch_size,
            drop_last=True
        )
        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def set_epoch(self, epoch):
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)

    def __len__(self):
        raise ValueError
    

class FastDataLoader:
    """DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch."""
    # def __init__(self, dataset, batch_size, num_workers):
    def __init__(self, dataset, batch_size, num_workers, rank, world_size, shuffle=True):
        super().__init__()

        self.sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )

        batch_sampler = torch.utils.data.BatchSampler(
            self.sampler,
            batch_size=batch_size,
            drop_last=False
        )

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

        self._length = len(batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length

    def set_epoch(self, epoch):
        """Sets the epoch for this sampler. Needed for proper shuffling."""
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)
