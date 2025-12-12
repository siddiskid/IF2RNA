

import os
import numpy as np
import pandas as pd
import h5py
import torch
from pathlib import Path
from torch.utils.data import Dataset, TensorDataset, Subset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from torchvision.transforms import Compose
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False
from joblib import Parallel, delayed


def load_labels(transcriptome_dataset):
    assert hasattr(transcriptome_dataset, 'transcriptomes'), \
        "Transcriptomes have not been loaded for this dataset"

    to_drop = ['Case.ID', 'Sample.ID', 'File.ID', 'Project.ID']
    df = transcriptome_dataset.transcriptomes.copy()
    patients = df['Case.ID'].values
    projects = df['Project.ID']
    df.drop(to_drop, axis=1, inplace=True)
    genes = df.columns
    df = np.log10(1 + df)
    y = df.values

    return y, genes, patients, projects


def load_and_aggregate_file(file, reduce=True):
    x = np.load(file)
    x = x[:, 3:]
    if reduce:
        x = np.mean(x, axis=0)
    else:
        x = np.concatenate((x, np.zeros((8000 - x.shape[0], 2048)))).transpose(1, 0)
    return x


def load_npy_data(file_list, reduce=True):
    X = np.array(Parallel(n_jobs=16)(delayed(load_and_aggregate_file)(file) 
                                     for file in tqdm(file_list)))
    return X


def make_dataset(dir, file_list, labels):
    images = []
    dir = os.path.expanduser(dir)

    for fname, label in zip(file_list, labels):
        path = os.path.join(dir, fname)
        if os.path.exists(path):
            item = (path, label)
            images.append(item)

    return images


class IF2RNADataset(TensorDataset):
    def __init__(self, genes, patients, projects, *tensors):
        super(IF2RNADataset, self).__init__(*tensors)
        self.genes = genes
        self.patients = patients
        self.projects = projects
        self.dim = 2048  # ResNet-50 feature dimension

    @classmethod
    def from_transcriptome_data(cls, transcriptome_dataset, tiles_path):
        y, cols, patients, projects = load_labels(transcriptome_dataset)

        file_list = [
            os.path.join(
                tiles_path, project.replace('-', '_'),
                '0.50_mpp', filename
            )
            for project, filename in transcriptome_dataset.metadata[['Project.ID', 'Slide.ID']].values
        ]
        X = load_npy_data(file_list)
        return cls(cols, patients, projects, torch.Tensor(X), torch.Tensor(y))


class ToTensor(object):
    def __init__(self, n_tiles=8000):
        self.n_tiles = n_tiles

    def __call__(self, sample):
        x = torch.from_numpy(sample).float()
        if x.shape[0] > self.n_tiles:
            x = x[:self.n_tiles]
        elif x.shape[0] < self.n_tiles:
            x = torch.cat((x, torch.zeros((self.n_tiles - x.shape[0], 2051))))
        return x.t()


class RemoveCoordinates(object):
    def __call__(self, sample):
        return sample[3:]


class IF2RNATileDataset(Dataset):
    def __init__(self, genes, patients, projects, projectname, file_list, labels,
                 tiles_root_path,
                 transform=None,
                 target_transform=None, 
                 masks=None):
        
        if transform is None:
            transform = Compose([ToTensor(), RemoveCoordinates()])
            
        samples = make_dataset(tiles_root_path, file_list, labels)
        if len(samples) == 0:
            raise(RuntimeError(f"Found 0 files in: {tiles_root_path}"))

        self.root = tiles_root_path
        self.patients = patients
        self.projects = projects
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform
        self.genes = genes
        self.dim = 2048
        self.masks = masks

    @classmethod
    def from_transcriptome_data(cls, transcriptome_dataset, tiles_root_path):
        projectname = transcriptome_dataset.projectname
        labels, cols, patients, projects = load_labels(transcriptome_dataset)
        file_list = [
            os.path.join(
                project.replace('-', '_'),
                '0.50_mpp', filename)
            for project, filename in transcriptome_dataset.metadata[['Project.ID', 'Slide.ID']].values
        ]
        return cls(cols, patients, projects, projectname, file_list, labels, tiles_root_path)

    def __getitem__(self, index):
        path, target = self.samples[index]
        if self.masks is not None:
            mask = self.masks[path.split('/')[-1]]
            idx = np.argsort(mask[:, 0])[::-1]
            sample = np.load(path)[idx] * mask[idx]
        else:
            sample = np.load(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)


class IFDataset(Dataset):
    
    def __init__(self, genes, patients, projects, if_features, labels):
        self.genes = genes
        self.patients = patients
        self.projects = projects
        self.if_features = if_features
        self.labels = labels
        self.n_channels = if_features.shape[1] if len(if_features.shape) > 2 else 1
    
    def __len__(self):
        return len(self.if_features)
    
    def __getitem__(self, idx):
        return self.if_features[idx], self.labels[idx]
    
    @classmethod
    def from_if_data(cls, if_images, gene_expression, genes, patients, projects):
        return cls(genes, patients, projects, if_images, gene_expression)


class IFTileTransform:
    
    def __init__(self, n_tiles=8000, normalize=True):
        self.n_tiles = n_tiles
        self.normalize = normalize
    
    def __call__(self, if_tile):
        if isinstance(if_tile, np.ndarray):
            if_tile = torch.from_numpy(if_tile).float()
        
        if len(if_tile.shape) == 3:
            if_tile = if_tile.permute(2, 0, 1)
        
        if self.normalize:
            for c in range(if_tile.shape[0]):
                channel = if_tile[c]
                if channel.max() > channel.min():
                    if_tile[c] = (channel - channel.min()) / (channel.max() - channel.min())
        
        return if_tile


def load_multichannel_image(file_path):
    file_path = Path(file_path) if not isinstance(file_path, Path) else file_path
    
    if file_path.suffix.lower() in ['.tif', '.tiff'] and TIFFFILE_AVAILABLE:
        img = tifffile.imread(file_path)
        if len(img.shape) == 3 and img.shape[0] < img.shape[1]:
            img = np.transpose(img, (1, 2, 0))
        elif len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
    else:
        img = np.array(Image.open(file_path))
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
    return img


def normalize_if_channels(img, method='percentile'):
    if method == 'percentile':
        for i in range(img.shape[-1]):
            p1, p99 = np.percentile(img[..., i], [1, 99])
            img[..., i] = np.clip((img[..., i] - p1) / (p99 - p1), 0, 1)
    elif method == 'minmax':
        for i in range(img.shape[-1]):
            min_val, max_val = img[..., i].min(), img[..., i].max()
            if max_val > min_val:
                img[..., i] = (img[..., i] - min_val) / (max_val - min_val)
    return img


def create_synthetic_if_data(n_samples=100, n_tiles=1000, n_channels=4, 
                           tile_size=224, n_genes=100):
    np.random.seed(42)
    torch.manual_seed(42)
    
    channel_names = ['DAPI', 'CD3', 'CD20', 'AF'] if n_channels == 4 else [f'Ch{i}' for i in range(n_channels)]
    
    X = []
    for _ in range(n_samples):
        sample_tiles = np.random.randn(2048, n_tiles)
        
        dapi_pattern = np.random.rand(n_tiles) * 0.3 + 0.7
        sample_tiles[:512] *= dapi_pattern.reshape(1, -1)
        
        if n_channels > 1:
            cd3_mask = np.random.rand(n_tiles) > 0.8
            sample_tiles[512:1024] *= cd3_mask.reshape(1, -1) * 2
        
        if n_channels > 2:
            cd20_mask = np.random.rand(n_tiles) > 0.85
            sample_tiles[1024:1536] *= cd20_mask.reshape(1, -1) * 1.5
        
        X.append(sample_tiles)
    
    X = np.array(X)
    X = torch.tensor(X, dtype=torch.float32)
    
    y = torch.randn(n_samples, n_genes) * 2 + 5
    y = torch.relu(y)
    
    immune_genes = n_genes // 4
    for i in range(immune_genes):
        if n_channels > 1:
            cd3_signal = torch.sum(X[:, 512:1024, :], dim=(1,2))
            y[:, i] += cd3_signal * 0.1
    
    patients = [f"IF_patient_{i:03d}" for i in range(n_samples)]
    projects = np.random.choice(['IF-BRCA', 'IF-LUNG', 'IF-COAD'], n_samples)
    
    return X, y, patients, projects


def create_synthetic_data(n_samples=100, n_tiles=1000, feature_dim=2048, n_genes=100):
    np.random.seed(42)
    torch.manual_seed(42)
    
    X = torch.randn(n_samples, feature_dim, n_tiles)
    
    y = torch.randn(n_samples, n_genes) * 2 + 5
    y = torch.relu(y)
    patients = [f"patient_{i:03d}" for i in range(n_samples)]
    projects = np.random.choice(['TCGA-BRCA', 'TCGA-LUAD', 'TCGA-COAD'], n_samples)
    
    return X, y, patients, projects
