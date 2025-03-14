import math
import numpy as np
from typing import Any, Iterator, Optional
import torch
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch.utils.data import Dataset, _DatasetKind
from torch.utils.data.distributed import DistributedSampler
from operator import itemgetter
import torch.distributed as dist
import warnings

from sklearn.cluster import KMeans
import numpy as np

__all__ = ['InfoBatch', 'SeTa', 'prune']


def info_hack_indices(self):
    with torch.autograd.profiler.record_function(self._profile_name):
        if self._sampler_iter is None:
            # TODO(https://github.com/pytorch/pytorch/issues/76750)
            self._reset()  # type: ignore[call-arg]
        if isinstance(self._dataset, InfoBatch):
            indices, data = self._next_data()
        else:
            data = self._next_data()
        self._num_yielded += 1
        if self._dataset_kind == _DatasetKind.Iterable and \
                self._IterableDataset_len_called is not None and \
                self._num_yielded > self._IterableDataset_len_called:
            warn_msg = ("Length of IterableDataset {} was reported to be {} (when accessing len(dataloader)), but {} "
                        "samples have been fetched. ").format(self._dataset, self._IterableDataset_len_called,
                                                                self._num_yielded)
            if self._num_workers > 0:
                warn_msg += ("For multiprocessing data-loading, this could be caused by not properly configuring the "
                                "IterableDataset replica at each worker. Please see "
                                "https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset for examples.")
            warnings.warn(warn_msg)
        if isinstance(self._dataset, InfoBatch):
            self._dataset.set_active_indices(indices)
        return data


_BaseDataLoaderIter.__next__ = info_hack_indices


@torch.no_grad()
def concat_all_gather(tensor, dim=0):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=dim)
    return output

import json
class Log:
    def __init__(self, path: str):
        self.path = path
        self.res = {}
    
    def add(self, key: str, value: Any):
        self.res[key] = value

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.res, f)

def prune(dataset, args):
    if args.prune_type == 'InfoBatch':
        dataset = InfoBatch(dataset, args.epochs, args.prune_ratio, args.delta)
        print(f'==> InfoBatch pruning: ratio={args.prune_ratio}')
    elif args.prune_type == 'SeTa':
        dataset = SeTa(dataset, args.epochs, args.prune_ratio,
                        args.num_group, args.window_scale,
                        args.delta)
        print(f'==> SeTa pruning: ratio={args.prune_ratio}, group={args.num_group}, window_scale={args.window_scale}')
    elif args.prune_type == 'Static':
        dataset = Static(dataset, args.epochs, args.prune_ratio)
        print(f'==> Static pruning: ratio={args.prune_ratio}')
    else:
        dataset = InfoBatch(dataset, args.epochs, 0.0, 0.0)
        print('==> No pruning')
    return dataset

class InfoBatch(Dataset):
    """
    InfoBatch aims to achieve lossless training speed up by randomly prunes a portion of less informative samples
    based on the loss distribution and rescales the gradients of the remaining samples to approximate the original
    gradient. See https://arxiv.org/pdf/2303.04947.pdf

    .. note::.
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for training.
        num_epochs (int): The number of epochs for pruning.
        prune_ratio (float, optional): The proportion of samples being pruned during training.
        delta (float, optional): The first delta * num_epochs the pruning process is conducted. It should be close to 1. Defaults to 0.875.
    """

    def __init__(self, dataset: Dataset, num_epochs: int,
                 prune_ratio: float = 0.5, delta: float = 0.875):
        self.dataset = dataset
        self.keep_ratio = min(1.0, max(1e-1, 1.0 - prune_ratio))
        self.num_epochs = num_epochs
        self.delta = delta
        # self.scores stores the loss value of each sample. Note that smaller value indicates the sample is better learned by the network.
        self.scores = torch.ones(len(self.dataset)) * 3
        self.weights = torch.ones(len(self.dataset))
        self.num_pruned_samples = 0
        self.cur_batch_index = None

    def set_active_indices(self, cur_batch_indices: torch.Tensor):
        self.cur_batch_index = cur_batch_indices

    def update(self, values):
        assert isinstance(values, torch.Tensor)
        batch_size = len(values)
        assert len(self.cur_batch_index) == batch_size, 'not enough index'
        device = values.device
        weights = self.weights[self.cur_batch_index].to(device)
        indices = self.cur_batch_index.to(device)
        loss_val = values.detach().clone()
        self.cur_batch_index = []

        if dist.is_available() and dist.is_initialized():
            iv = torch.cat([indices.view(1, -1), loss_val.view(1, -1)], dim=0)
            iv_whole_group = concat_all_gather(iv, 1)
            indices = iv_whole_group[0]
            loss_val = iv_whole_group[1]
        self.scores[indices.cpu().long()] = loss_val.cpu()
        values.mul_(weights)
        return values.mean()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # self.cur_batch_index.append(index)
        return index, self.dataset[index] # , index
        # return self.dataset[index], index, self.scores[index]
    
    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def prune(self):
        # Prune samples that are well learned, rebalance the weight by scaling up remaining
        # well learned samples' learning rate to keep estimation about the same
        # for the next version, also consider new class balance

        well_learned_mask = (self.scores < self.scores.mean()).numpy()
        well_learned_indices = np.where(well_learned_mask)[0]
        remained_indices = np.where(~well_learned_mask)[0].tolist()
        # print('#well learned samples %d, #remained samples %d, len(dataset) = %d' % (np.sum(well_learned_mask), np.sum(~well_learned_mask), len(self.dataset)))
        selected_indices = np.random.choice(well_learned_indices, int(
            self.keep_ratio * len(well_learned_indices)), replace=False)
        self.reset_weights()
        if len(selected_indices) > 0:
            self.weights[selected_indices] = 1 / self.keep_ratio
            remained_indices.extend(selected_indices)
        self.num_pruned_samples += len(self.dataset) - len(remained_indices)
        np.random.shuffle(remained_indices)
        
        saved = 1 - len(remained_indices) / len(self.dataset)
        print(f'\n|--| #sampled: {len(remained_indices)} #saved: {saved * 100:.2f}%')
        
        return remained_indices

    @property
    def sampler(self):
        sampler = IBSampler(self)
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedIBSampler(sampler)
        return sampler

    def no_prune(self):
        samples_indices = list(range(len(self)))
        np.random.shuffle(samples_indices)
        print(f'|--| #sampled: {len(samples_indices)} #saved: {0.0 * 100:.2f}%')
        return samples_indices

    def mean_score(self):
        return self.scores.mean()

    def get_weights(self, indexes):
        return self.weights[indexes]

    def get_pruned_count(self):
        return self.num_pruned_samples
    
    def get_saved_ratio(self):
        return self.num_pruned_samples / (len(self.dataset) * self.num_epochs)

    @property
    def stop_prune(self):
        return self.num_epochs * self.delta

    def reset_weights(self):
        self.weights[:] = 1

class Static(InfoBatch):
    def __init__(
        self,
        dataset,
        num_epoch,
        prune_ratio = 0.5):
        num = int((1 - prune_ratio) * len(dataset))
        dataset = torch.utils.data.Subset(dataset, list(range(num)))
        super(Static, self).__init__(dataset, num_epoch, 0, 1)

# =============> SeTa
def random_select_with_ratio(data, ratio):
    assert 0 <= ratio <= 1
    assert len(data) > 0

    indices = torch.arange(len(data))
    num = int(len(data) * ratio)
    _indices = torch.randperm(len(data))[:num]
    selected_indices = indices[_indices]
    selected_data = data[_indices]
    return selected_data, selected_indices

def kmeans_group(scores, indices, num_group=10):
    assert len(scores) > 0, "empty scores"
    assert len(scores) > num_group, "num_group must be less than len(scores)"
    if num_group == 1:
        return [scores], [indices]

    max_score, min_score = scores.max(), scores.min()
    if max_score == min_score:
        return torch.chunk(scores, num_group), \
            torch.chunk(indices, num_group)
    
    kmeans = KMeans(n_clusters=num_group, random_state=0
                    ).fit(scores.unsqueeze(1))
    labels = kmeans.labels_
    
    grouped_scores = [[] for _ in range(num_group)]
    grouped_indices = [[] for _ in range(num_group)]

    for score, index, label in zip(scores, indices, labels):
        grouped_scores[label].append(score)
        grouped_indices[label].append(index)

    group_centers = [np.mean(group) for group in grouped_scores]
    sorted_groups = sorted(zip(group_centers, grouped_scores, grouped_indices), key=lambda x: x[0])

    sorted_grouped_scores = [group[1] for group in sorted_groups]
    sorted_grouped_indices = [group[2] for group in sorted_groups]
    return sorted_grouped_scores, sorted_grouped_indices

def slide_easy2hard(grouped_indices, cur_iterations, window_scale=0.5):
    if cur_iterations == 0:
        return grouped_indices
    num_group = len(grouped_indices)
    window_size = round(num_group * window_scale)
    assert window_size > 0
    slide_size = num_group - window_size
    
    start = cur_iterations % (slide_size + 1)
    end = start + window_size
    return grouped_indices[start: end]


class SeTa(InfoBatch):
    """
    """
    def __init__(
        self,
        dataset: Dataset,
        num_epochs: int,
        prune_ratio: float = 0.0,
        num_group: int = 10,
        window_scale: float = 0.5,
        delta: float = 0.875,
    ):
        super(SeTa, self).__init__(dataset, num_epochs, prune_ratio, delta)
        self.num_group = num_group
        self.window_scale = window_scale
        self.iterations = 0

    def prune(self):
        # Synchronize random state across processes
        if dist.is_available() and dist.is_initialized():
            seed = torch.tensor(self.iterations, dtype=torch.int64).cuda()
            dist.broadcast(seed, src=0)
            torch.manual_seed(seed.item())
            np.random.seed(seed.item())

        # 1. randomly select samples with keep_ratio
        scores, indices = random_select_with_ratio(self.scores, self.keep_ratio)

        # 2. group samples
        grouped_scores, grouped_indices = kmeans_group(
            scores, indices,
            num_group=self.num_group,
        )
        
        # 3. selection samples with sliding from easy to hard
        selected_grouped_indices = slide_easy2hard(grouped_indices, self.iterations, self.window_scale)
        selected_indices = [index for group in selected_grouped_indices
                            for index in group]
        self.iterations += 1

        if len(selected_indices) == 0:
            # avoid empty indices
            selected_indices = np.random.choice(len(self.dataset), 1)
        
        # print info
        raw_each_group_size = [len(indices) for indices in grouped_indices]
        rel_each_group_size = [len(indices) for indices in selected_grouped_indices]
        self.print(raw_each_group_size, rel_each_group_size, selected_indices)

        # count num of pruned samples
        self.num_pruned_samples += len(self.dataset) - len(selected_indices)

        np.random.shuffle(selected_indices)
        return np.array(selected_indices)
    
    def no_prune(self):
        # partially annealing
        sample_indices = super().no_prune()
        size = int(self.keep_ratio * len(sample_indices))
        selected_indices = np.random.choice(sample_indices, size, replace=False)
        
        self.num_pruned_samples += len(self.dataset) - len(selected_indices)
        self.print(None, None, selected_indices)
        return np.array(selected_indices)
    
    def print(self, group_size, each_group_size, sampled_indices):
        print('\n')
        print(f'|--| each group size: {group_size}')
        print(f'|--| selected each group size: {each_group_size}')

        saved = 1 - len(sampled_indices) / len(self.dataset)
        print(f'|--| #sampled: {len(sampled_indices)} #saved: {saved * 100:.2f}%')

# <============= SeTa

class IBSampler(object):
    def __init__(self, dataset: InfoBatch):
        self.dataset = dataset
        self.stop_prune = dataset.stop_prune
        self.iterations = 1
        self.iter_obj = None
        # self.reset()
        self.sample_indices = list(range(len(self.dataset)))

    def __getitem__(self, idx):
        return self.sample_indices[idx]

    def reset(self):
        np.random.seed(self.iterations)
        if self.iterations > self.stop_prune:
            # print('we are going to stop prune, #stop prune %d, #cur iterations %d' % (self.iterations, self.stop_prune))
            if self.iterations == self.stop_prune + 1:
                self.dataset.reset_weights()
            self.sample_indices = self.dataset.no_prune()
        else:
            # print('we are going to continue pruning, #stop prune %d, #cur iterations %d' % (self.iterations, self.stop_prune))
            self.sample_indices = self.dataset.prune()
        if self.iterations == self.dataset.num_epochs:
            print("="*20, "final", "="*20)
            print("===> saved_ratio: ", self.dataset.get_saved_ratio())
            print("===> pruned_count: ", self.dataset.get_pruned_count())
            print("="*20, "final", "="*20)
        self.iter_obj = iter(self.sample_indices)
        self.iterations += 1

    def __next__(self):
        return next(self.iter_obj) # may raise StopIteration
        
    def __len__(self):
        return len(self.sample_indices)

    def __iter__(self):
        self.reset()
        return self


class DistributedIBSampler(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler can change size during training.
    """
    class DatasetFromSampler(Dataset):
        def __init__(self, sampler: IBSampler):
            self.dataset = sampler
            # self.indices = None
            print("Use DistributedIBSampler")
 
        def reset(self, ):
            self.indices = None
            self.dataset.reset()

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index: int):
            """Gets element of the dataset.
            Args:
                index: index of the element in the dataset
            Returns:
                Single element by index
            """
            # if self.indices is None:
            #    self.indices = list(self.dataset)
            return self.dataset[index]

    def __init__(self, dataset: IBSampler, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = True) -> None:
        sampler = self.DatasetFromSampler(dataset)
        super(DistributedIBSampler, self).__init__(
            sampler, num_replicas, rank, shuffle, seed, drop_last)
        self.sampler = sampler
        self.dataset = sampler.dataset.dataset # the real dataset.
        self.iter_obj = None

    def __iter__(self) -> Iterator[int]:
        """
        Notes self.dataset is actually an instance of IBSampler rather than InfoBatch.
        """
        self.sampler.reset()
        if self.drop_last and len(self.sampler) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.sampler) - self.num_replicas) /
                self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(
                len(self.sampler) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # type: ignore[arg-type]
            indices = torch.randperm(len(self.sampler), generator=g).tolist()
        else:
            indices = list(range(len(self.sampler)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size /
                            len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        # print('distribute iter is called')
        self.iter_obj = iter(itemgetter(*indices)(self.sampler))
        return self.iter_obj
   
