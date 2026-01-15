

import numpy as np
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
from scipy.ndimage import gaussian_filter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimulatedIFGenerator:
    
    def __init__(self, image_size=224, seed=42, n_channels=50):
        self.image_size = image_size
        self.seed = seed
        self.n_channels = n_channels
        
        if n_channels == 6:
            self.channel_names = ['DAPI', 'CD3', 'CD20', 'CD45', 'CD68', 'CK']
        elif n_channels == 50:
            self.channel_names = self._get_50_channel_panel()
        else:
            self.channel_names = [f'Marker_{i:02d}' for i in range(n_channels)]
            
        self._setup_marker_properties()
    
    def _get_50_channel_panel(self):
        return [
            'DAPI', 'Hoechst', 'PI', 'PCNA', 'Ki67',
            'CD3', 'CD4', 'CD8', 'FOXP3', 'CD25', 'PD1', 'CTLA4', 'CD45RO',
            'CD20', 'CD19', 'CD79a', 'BCL6',
            'CD68', 'CD163', 'CD11b', 'CD11c', 'CD14', 'iNOS',
            'CK', 'EpCAM', 'E-Cadherin', 'CK7', 'CK19', 'Beta-Catenin',
            'PDL1', 'LAG3', 'TIM3', 'TIGIT',
            'CD31', 'VEGF', 'CD34', 'vWF',
            'Vimentin', 'SMA', 'FAP', 'Collagen_I', 'Collagen_IV',
            'Caspase3', 'PARP', 'p53', 'MDM2',
            'HER2', 'EGFR', 'c-MYC', 'BCL2'
        ]
    
    def _setup_marker_properties(self):
        self.marker_properties = {}
        
        for i, marker in enumerate(self.channel_names):
            props = {
                'base_intensity': 0.3,
                'positive_fraction': 0.1,
                'spatial_clustering': False,
                'cell_type': 'general'
            }
            
            if marker in ['DAPI', 'Hoechst', 'PI']:
                props.update({'base_intensity': 0.8, 'positive_fraction': 0.95, 'cell_type': 'nuclear'})
            elif marker in ['CD3', 'CD4', 'CD8', 'FOXP3', 'CD25', 'PD1', 'CTLA4', 'CD45RO']:
                props.update({'base_intensity': 0.6, 'positive_fraction': 0.15, 'spatial_clustering': True, 'cell_type': 'T_cell'})
            elif marker in ['CD20', 'CD19', 'CD79a', 'BCL6']:
                props.update({'base_intensity': 0.5, 'positive_fraction': 0.08, 'spatial_clustering': True, 'cell_type': 'B_cell'})
            elif marker in ['CD68', 'CD163', 'CD11b', 'CD11c', 'CD14', 'iNOS']:
                props.update({'base_intensity': 0.4, 'positive_fraction': 0.12, 'cell_type': 'macrophage'})
            elif marker in ['CK', 'EpCAM', 'E-Cadherin', 'CK7', 'CK19', 'Beta-Catenin']:
                props.update({'base_intensity': 0.7, 'positive_fraction': 0.4, 'cell_type': 'epithelial'})
            elif marker in ['PDL1', 'LAG3', 'TIM3', 'TIGIT']:
                props.update({'base_intensity': 0.3, 'positive_fraction': 0.05, 'cell_type': 'checkpoint'})
            elif marker in ['CD31', 'VEGF', 'CD34', 'vWF']:
                props.update({'base_intensity': 0.6, 'positive_fraction': 0.03, 'cell_type': 'endothelial'})
            elif marker in ['Vimentin', 'SMA', 'FAP', 'Collagen_I', 'Collagen_IV']:
                props.update({'base_intensity': 0.4, 'positive_fraction': 0.2, 'cell_type': 'stromal'})
            elif marker in ['Ki67', 'PCNA']:
                props.update({'base_intensity': 0.5, 'positive_fraction': 0.1, 'cell_type': 'proliferation'})
            elif marker in ['Caspase3', 'PARP', 'p53', 'MDM2']:
                props.update({'base_intensity': 0.3, 'positive_fraction': 0.05, 'cell_type': 'apoptosis'})
                
            self.marker_properties[marker] = props
        
    def _create_cellular_structure(self, cell_density=0.3, seed_offset=0):
        np.random.seed(self.seed + seed_offset)
        
        n_cells = int(cell_density * self.image_size ** 2 / 100)
        cell_centers_x = np.random.randint(0, self.image_size, n_cells)
        cell_centers_y = np.random.randint(0, self.image_size, n_cells)
        
        dapi = np.zeros((self.image_size, self.image_size))
        
        for cx, cy in zip(cell_centers_x, cell_centers_y):
            y, x = np.ogrid[0:self.image_size, 0:self.image_size]
            nucleus_size = np.random.uniform(3, 6)
            nucleus_intensity = np.random.uniform(0.6, 1.0)
            
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            nucleus = nucleus_intensity * np.exp(-dist**2 / (2 * nucleus_size**2))
            dapi += nucleus
        
        dapi = gaussian_filter(dapi, sigma=1.0)
        dapi += np.random.normal(0, 0.02, dapi.shape)
        dapi = np.clip(dapi, 0, 1)
        
        return dapi, (cell_centers_x, cell_centers_y)
    
    def _create_marker_channel(self, cell_centers, base_intensity, 
                               positive_fraction, spatial_clustering=False):
        cell_centers_x, cell_centers_y = cell_centers
        n_cells = len(cell_centers_x)
        
        if spatial_clustering:
            n_hotspots = np.random.randint(1, 4)
            hotspot_centers = np.random.randint(0, self.image_size, (n_hotspots, 2))
            
            distances = []
            for cx, cy in zip(cell_centers_x, cell_centers_y):
                min_dist = min(np.sqrt((cx - hx)**2 + (cy - hy)**2) 
                              for hx, hy in hotspot_centers)
                distances.append(min_dist)
            
            distances = np.array(distances)
            probs = np.exp(-distances / 50) 
            probs = probs / probs.max() * positive_fraction * 2
            positive_mask = np.random.random(n_cells) < probs
        else:
            positive_mask = np.random.random(n_cells) < positive_fraction
        
        channel = np.zeros((self.image_size, self.image_size))
        
        for i, (cx, cy) in enumerate(zip(cell_centers_x, cell_centers_y)):
            if positive_mask[i]:
                y, x = np.ogrid[0:self.image_size, 0:self.image_size]
                cell_size = np.random.uniform(4, 8)
                intensity = base_intensity * np.random.uniform(0.7, 1.3)
                
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                blob = intensity * np.exp(-dist**2 / (2 * cell_size**2))
                channel += blob
        
        channel = gaussian_filter(channel, sigma=1.2)
        channel += np.random.normal(0, 0.01, channel.shape)
        channel = np.clip(channel, 0, 1)
        
        return channel
    
    def generate_tumor_region(self, seed_offset=0):
        # High cell density in tumor
        dapi, cell_centers = self._create_cellular_structure(
            cell_density=0.5, seed_offset=seed_offset
        )
        
        channels = []
        
        for i, marker in enumerate(self.channel_names):
            if marker in ['DAPI', 'Hoechst', 'PI'] and i == 0:
                channels.append(dapi)
            else:
                props = self.marker_properties[marker].copy()
                
                if props['cell_type'] == 'epithelial':
                    props['positive_fraction'] *= 3.0
                    props['base_intensity'] *= 1.2
                elif props['cell_type'] in ['T_cell', 'B_cell']:
                    props['positive_fraction'] *= 0.3
                    props['spatial_clustering'] = True
                elif props['cell_type'] == 'macrophage':
                    props['positive_fraction'] *= 0.8
                elif props['cell_type'] == 'proliferation':
                    props['positive_fraction'] *= 2.0
                elif props['cell_type'] == 'stromal':
                    props['positive_fraction'] *= 0.5
                props['positive_fraction'] = min(props['positive_fraction'], 0.9)
                props['base_intensity'] = min(props['base_intensity'], 1.0)
                
                channel = self._create_marker_channel(
                    cell_centers, 
                    props['base_intensity'],
                    props['positive_fraction'],
                    props['spatial_clustering']
                )
                channels.append(channel)
        
        return np.stack(channels, axis=0)
    
    def generate_immune_aggregate(self, seed_offset=0):
        dapi, cell_centers = self._create_cellular_structure(
            cell_density=0.6, seed_offset=seed_offset
        )
        
        channels = []
        
        # Generate all channels based on tissue-specific patterns
        for i, marker in enumerate(self.channel_names):
            if marker in ['DAPI', 'Hoechst', 'PI'] and i == 0:
                channels.append(dapi)
            else:
                props = self.marker_properties[marker].copy()
                
                if props['cell_type'] in ['T_cell', 'B_cell']:
                    props['positive_fraction'] *= 4.0
                    props['base_intensity'] *= 1.3
                    props['spatial_clustering'] = True
                elif props['cell_type'] == 'macrophage':
                    props['positive_fraction'] *= 2.5
                    props['spatial_clustering'] = True
                elif props['cell_type'] == 'epithelial':
                    props['positive_fraction'] *= 0.1
                elif props['cell_type'] == 'checkpoint':
                    props['positive_fraction'] *= 3.0
                elif props['cell_type'] == 'proliferation':
                    props['positive_fraction'] *= 1.5
                elif props['cell_type'] == 'stromal':
                    props['positive_fraction'] *= 0.3
                
                # Clamp values
                props['positive_fraction'] = min(props['positive_fraction'], 0.9)
                props['base_intensity'] = min(props['base_intensity'], 1.0)
                
                channel = self._create_marker_channel(
                    cell_centers,
                    props['base_intensity'], 
                    props['positive_fraction'],
                    props['spatial_clustering']
                )
                channels.append(channel)
        
        return np.stack(channels, axis=0)
    
    def generate_stroma_region(self, seed_offset=0):
        dapi, cell_centers = self._create_cellular_structure(
            cell_density=0.2, seed_offset=seed_offset
        )
        
        channels = []
        
        for i, marker in enumerate(self.channel_names):
            if marker in ['DAPI', 'Hoechst', 'PI'] and i == 0:
                channels.append(dapi)
            else:
                props = self.marker_properties[marker].copy()
                
                if props['cell_type'] == 'stromal':
                    props['positive_fraction'] *= 3.0
                    props['base_intensity'] *= 1.2
                elif props['cell_type'] in ['T_cell', 'B_cell']:
                    props['positive_fraction'] *= 0.4
                elif props['cell_type'] == 'epithelial':
                    props['positive_fraction'] *= 0.05
                elif props['cell_type'] == 'macrophage':
                    props['positive_fraction'] *= 0.6
                elif props['cell_type'] == 'endothelial':
                    props['positive_fraction'] *= 1.5
                elif props['cell_type'] == 'proliferation':
                    props['positive_fraction'] *= 0.2
                
                props['positive_fraction'] = min(props['positive_fraction'], 0.9)
                props['base_intensity'] = min(props['base_intensity'], 1.0)
                
                channel = self._create_marker_channel(
                    cell_centers,
                    props['base_intensity'],
                    props['positive_fraction'], 
                    props['spatial_clustering']
                )
                channels.append(channel)
        
        return np.stack(channels, axis=0)
    
    def generate_normal_region(self, seed_offset=0):
        dapi, cell_centers = self._create_cellular_structure(
            cell_density=0.4, seed_offset=seed_offset
        )
        
        channels = []
        
        for i, marker in enumerate(self.channel_names):
            if marker in ['DAPI', 'Hoechst', 'PI'] and i == 0:
                channels.append(dapi)
            else:
                props = self.marker_properties[marker].copy()
                
                if props['cell_type'] == 'epithelial':
                    props['positive_fraction'] *= 1.5
                elif props['cell_type'] in ['T_cell', 'B_cell']:
                    props['positive_fraction'] *= 0.6
                elif props['cell_type'] == 'macrophage':
                    props['positive_fraction'] *= 0.5
                elif props['cell_type'] == 'stromal':
                    props['positive_fraction'] *= 0.8
                elif props['cell_type'] == 'proliferation':
                    props['positive_fraction'] *= 0.3
                elif props['cell_type'] == 'checkpoint':
                    props['positive_fraction'] *= 0.2
                elif props['cell_type'] == 'endothelial':
                    props['positive_fraction'] *= 0.8
                
                props['positive_fraction'] = min(props['positive_fraction'], 0.9)
                props['base_intensity'] = min(props['base_intensity'], 1.0)
                
                channel = self._create_marker_channel(
                    cell_centers,
                    props['base_intensity'],
                    props['positive_fraction'],
                    props['spatial_clustering']
                )
                channels.append(channel)
        
        return np.stack(channels, axis=0)
    
    def generate_for_tissue_type(self, tissue_type, seed_offset=0):
        tissue_type = tissue_type.strip()
        
        if tissue_type == 'Tumor':
            return self.generate_tumor_region(seed_offset)
        elif tissue_type == 'Immune_Aggregate':
            return self.generate_immune_aggregate(seed_offset)
        elif tissue_type == 'Stroma':
            return self.generate_stroma_region(seed_offset)
        elif tissue_type == 'Normal':
            return self.generate_normal_region(seed_offset)
        else:
            logger.warning(f"Unknown tissue type '{tissue_type}', using Tumor")
            return self.generate_tumor_region(seed_offset)
    
    def generate_batch(self, tissue_types, return_tensor=True):
        images = []
        
        for i, tissue_type in enumerate(tissue_types):
            img = self.generate_for_tissue_type(tissue_type, seed_offset=i)
            images.append(img)
        
        batch = np.stack(images, axis=0)
        
        if return_tensor and TORCH_AVAILABLE:
            batch = torch.from_numpy(batch).float()
        elif return_tensor and not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, returning numpy array")
        
        logger.info(f"Generated batch of {len(tissue_types)} IF images with shape {batch.shape}")
        
        return batch


def test_generator():
    print("Testing IF generator")
    
    generator = SimulatedIFGenerator(image_size=224, seed=42)
    
    tissue_types = ['Tumor', 'Immune_Aggregate', 'Stroma', 'Normal']
    
    for tissue in tissue_types:
        img = generator.generate_for_tissue_type(tissue, seed_offset=0)
        print(f"{tissue}: {img.shape}")
    
    test_types = ['Tumor', 'Tumor', 'Immune_Aggregate', 'Stroma', 'Normal']
    batch = generator.generate_batch(test_types, return_tensor=True)
    print(f"Batch: {batch.shape}")


# Factory functions for different configurations

def create_basic_if_generator(image_size=224, seed=42):
    return SimulatedIFGenerator(image_size=image_size, seed=seed, n_channels=6)


def create_rosie_compatible_if_generator(image_size=224, seed=42):
    return SimulatedIFGenerator(image_size=image_size, seed=seed, n_channels=50)


def create_if_generator(n_channels=50, image_size=224, seed=42):
    return SimulatedIFGenerator(image_size=image_size, seed=seed, n_channels=n_channels)
    print(f"Range: {batch.min():.3f} - {batch.max():.3f}")
    
    return generator


if __name__ == '__main__':
    generator = test_generator()
