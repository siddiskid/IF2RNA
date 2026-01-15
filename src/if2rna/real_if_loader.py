

import numpy as np
import pandas as pd
from pathlib import Path
import logging
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealIFImageLoader:
    
    def __init__(self, data_dir, image_size=224):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.dcc_files = []
        self.pkc_file = None
        self.channel_names = []
        self.if_data_cache = {}
        
        self._discover_files()
        self._load_probe_config()
        
    def _discover_files(self):
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            return
        
        self.dcc_files = sorted(list(self.data_dir.glob("*.dcc")))
        pkc_files = list(self.data_dir.glob("*.pkc"))
        
        if pkc_files:
            self.pkc_file = pkc_files[0]
        
        logger.info(f"Found {len(self.dcc_files)} DCC files")
        if self.pkc_file:
            logger.info(f"Found PKC file: {self.pkc_file.name}")
    
    def _load_probe_config(self):
        if not self.pkc_file or not self.pkc_file.exists():
            logger.warning("No PKC file found, using default channel names")
            self.channel_names = [f"Marker_{i:02d}" for i in range(50)]
            return
        
        try:
            with open(self.pkc_file, 'r') as f:
                lines = f.readlines()
            
            in_targets = False
            for line in lines:
                line = line.strip()
                if line.startswith('<Name>'):
                    if in_targets:
                        name = line.replace('<Name>', '').replace('</Name>', '').strip()
                        if name and name not in self.channel_names:
                            self.channel_names.append(name)
                
                if '<Targets>' in line:
                    in_targets = True
                elif '</Targets>' in line:
                    in_targets = False
            
            if not self.channel_names:
                self.channel_names = [f"Marker_{i:02d}" for i in range(50)]
            
            logger.info(f"Loaded {len(self.channel_names)} channel names from PKC")
            
        except Exception as e:
            logger.warning(f"Could not parse PKC file: {e}")
            self.channel_names = [f"Marker_{i:02d}" for i in range(50)]
    
    def _load_dcc_file(self, dcc_path):
        try:
            with open(dcc_path, 'r') as f:
                lines = f.readlines()
            
            counts = {}
            in_code_summary = False
            
            for line in lines:
                line = line.strip()
                
                if '<Code_Summary>' in line:
                    in_code_summary = True
                    continue
                elif '</Code_Summary>' in line:
                    in_code_summary = False
                    continue
                
                if in_code_summary and '\t' in line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        gene_name = parts[0]
                        try:
                            count = float(parts[1])
                            counts[gene_name] = count
                        except ValueError:
                            continue
            
            return counts
            
        except Exception as e:
            logger.error(f"Error loading DCC file {dcc_path}: {e}")
            return {}
    
    def load_all_dcc_data(self):
        all_data = []
        roi_names = []
        
        for dcc_file in self.dcc_files:
            counts = self._load_dcc_file(dcc_file)
            if counts:
                all_data.append(counts)
                roi_names.append(dcc_file.stem)
        
        if not all_data:
            logger.warning("No DCC data loaded")
            return None, None
        
        df = pd.DataFrame(all_data, index=roi_names)
        df = df.fillna(0)
        
        logger.info(f"Loaded {len(df)} ROIs with {len(df.columns)} markers")
        
        return df, roi_names
    
    def _generate_if_image_from_counts(self, counts_dict, tissue_type='Unknown', seed_offset=0):
        np.random.seed(42 + seed_offset)
        
        if_channels = []
        
        for channel_name in self.channel_names[:50]:
            intensity = counts_dict.get(channel_name, 0)
            
            normalized_intensity = np.log1p(intensity) / 10.0
            normalized_intensity = np.clip(normalized_intensity, 0, 1)
            
            channel = self._create_spatial_pattern(
                normalized_intensity, 
                tissue_type, 
                seed_offset
            )
            if_channels.append(channel)
        
        while len(if_channels) < 50:
            if_channels.append(np.zeros((self.image_size, self.image_size)))
        
        return np.stack(if_channels[:50], axis=0)
    
    def _create_spatial_pattern(self, intensity, tissue_type, seed_offset):
        np.random.seed(42 + seed_offset)
        
        n_spots = int(intensity * 100) + 1
        
        channel = np.zeros((self.image_size, self.image_size))
        
        for _ in range(n_spots):
            cx = np.random.randint(0, self.image_size)
            cy = np.random.randint(0, self.image_size)
            
            y, x = np.ogrid[0:self.image_size, 0:self.image_size]
            spot_size = np.random.uniform(3, 7)
            spot_intensity = intensity * np.random.uniform(0.7, 1.0)
            
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            spot = spot_intensity * np.exp(-dist**2 / (2 * spot_size**2))
            channel += spot
        
        from scipy.ndimage import gaussian_filter
        channel = gaussian_filter(channel, sigma=1.0)
        
        channel += np.random.normal(0, 0.01, channel.shape)
        channel = np.clip(channel, 0, 1)
        
        return channel
    
    def generate_for_roi(self, roi_index, tissue_type='Unknown', seed_offset=0):
        if roi_index >= len(self.dcc_files):
            logger.error(f"ROI index {roi_index} out of range")
            return None
        
        dcc_file = self.dcc_files[roi_index]
        
        if str(dcc_file) in self.if_data_cache:
            counts = self.if_data_cache[str(dcc_file)]
        else:
            counts = self._load_dcc_file(dcc_file)
            self.if_data_cache[str(dcc_file)] = counts
        
        if_image = self._generate_if_image_from_counts(counts, tissue_type, seed_offset)
        
        return if_image
    
    def generate_for_tissue_type(self, tissue_type, roi_index=0, seed_offset=0):
        return self.generate_for_roi(roi_index, tissue_type, seed_offset)
    
    def generate_batch(self, roi_indices, tissue_types=None, return_tensor=True):
        if tissue_types is None:
            tissue_types = ['Unknown'] * len(roi_indices)
        
        images = []
        for i, roi_idx in enumerate(roi_indices):
            tissue_type = tissue_types[i] if i < len(tissue_types) else 'Unknown'
            img = self.generate_for_roi(roi_idx, tissue_type, seed_offset=i)
            if img is not None:
                images.append(img)
        
        if not images:
            logger.error("No images generated")
            return None
        
        batch = np.stack(images, axis=0)
        
        if return_tensor and TORCH_AVAILABLE:
            batch = torch.from_numpy(batch).float()
        elif return_tensor and not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, returning numpy array")
        
        logger.info(f"Generated batch of {len(images)} IF images with shape {batch.shape}")
        
        return batch
    
    def get_n_channels(self):
        return min(len(self.channel_names), 50)
    
    def get_channel_names(self):
        return self.channel_names[:50]


def create_real_if_loader(data_dir, image_size=224):
    return RealIFImageLoader(data_dir=data_dir, image_size=image_size)


def test_real_if_loader():
    print("Testing RealIFImageLoader")
    
    data_dir = Path(__file__).parent.parent.parent / "data" / "real_geomx"
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    loader = RealIFImageLoader(data_dir)
    
    print(f"Found {len(loader.dcc_files)} DCC files")
    print(f"Channel names: {loader.channel_names[:5]}...")
    
    dcc_data, roi_names = loader.load_all_dcc_data()
    if dcc_data is not None:
        print(f"Loaded data shape: {dcc_data.shape}")
        print(f"ROI names: {roi_names}")
    
    if len(loader.dcc_files) > 0:
        print("\nGenerating IF image for first ROI...")
        if_image = loader.generate_for_roi(0, tissue_type='Tumor', seed_offset=0)
        print(f"Generated IF image shape: {if_image.shape}")
        print(f"Value range: {if_image.min():.3f} - {if_image.max():.3f}")
        
        print("\nGenerating batch...")
        batch = loader.generate_batch([0, 1], tissue_types=['Tumor', 'Stroma'])
        if batch is not None:
            print(f"Batch shape: {batch.shape}")
    
    print("RealIFImageLoader test complete")


if __name__ == '__main__':
    test_real_if_loader()
