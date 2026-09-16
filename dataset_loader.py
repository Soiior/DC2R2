import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


def load_dataset(args):
    data, targets = load_mat(args)
    dataset_train = TrainDatasetImproved(
        data=data,
        missing_rate=args.missing_rate,
    )
    dataset_test = TestDataset(data, targets)

    return dataset_train, dataset_test


def load_mat(args):
    data_X = []
    label_y = None
    if args.dataset == 'SearchSnippets':
        mat = sio.loadmat(os.path.join(args.data_path, 'SearchSnippets.mat'))  
        data_X = mat['X'] 
        label_y = np.squeeze(mat['Y'])
        print(f"SearchSnippets data shape: {data_X.shape}, labels shape: {label_y.shape}")

    elif args.dataset == 'AgNews':
        mat = sio.loadmat(os.path.join(args.data_path, 'AgNews.mat'))  
        data_X = mat['X'] 
        label_y = np.squeeze(mat['Y'])
        print(f"AgNews data shape: {data_X.shape}, labels shape: {label_y.shape}")

    elif args.dataset == 'Biomedical':
        mat = sio.loadmat(os.path.join(args.data_path, 'Biomedical.mat'))  
        data_X = mat['X'] 
        label_y = np.squeeze(mat['Y'])
        print(f"Biomedical data shape: {data_X.shape}, labels shape: {label_y.shape}")

    elif args.dataset == 'StackOverflow':
        mat = sio.loadmat(os.path.join(args.data_path, 'StackOverflow.mat'))  
        data_X = mat['X'] 
        label_y = np.squeeze(mat['Y'])
        print(f"StackOverflow data shape: {data_X.shape}, labels shape: {label_y.shape}")

    else:
        raise ValueError('Unknown Dataset')

    return data_X, label_y

class TrainDatasetImproved(Dataset):
    def __init__(self, data: np.ndarray, missing_rate: float = 0):
        self.data = torch.FloatTensor(data)
        self.n_samples, self.n_features = data.shape
        self.missing_rate = missing_rate
        self.n_views = 2

    def _generate_sample_masks(self) -> torch.Tensor:
        n_missing = int(self.n_features * self.missing_rate)
        if n_missing == 0:
            return torch.ones(self.n_views, self.n_features)

        first_missing = np.random.choice(
            self.n_features, size=n_missing, replace=False)
        remaining = np.setdiff1d(np.arange(self.n_features), first_missing)
        second_missing = np.random.choice(
            remaining, size=min(n_missing, len(remaining)), replace=False)

        masks = []
        for missing_indices in (first_missing, second_missing):
            mask = torch.ones(self.n_features)
            mask[missing_indices] = 0
            masks.append(mask)
        return torch.stack(masks)
    
    def __getitem__(self, idx: int):
        data = self.data[idx]
        masks = self._generate_sample_masks()
        views_data = [data * masks[view_idx] for view_idx in range(self.n_views)]
        return views_data, data.clone()
    
    def __len__(self) -> int:
        return self.n_samples


class TestDataset(Dataset):
    def __init__(self, data_X, label_y):
        super(TestDataset, self).__init__()
        self.data = data_X
        self.targets = label_y - np.min(label_y)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        feat_tensor = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.targets[idx], dtype=torch.long)
        return idx, feat_tensor, label
