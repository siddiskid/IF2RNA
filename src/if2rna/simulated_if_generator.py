"""
Simulated IF Image Generator for IF2RNA
Generates realistic multi-channel immunofluorescence images based on tissue type
"""

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
    """
    Generates realistic multi-channel IF images with tissue-specific patterns.
    
    Supports both 6-channel (basic) and 50-channel (ROSIE-compatible) modes.
    """
    
    def __init__(self, image_size=224, seed=42, n_channels=50):
        """
        Args:
            image_size: Size of generated square images
            seed: Random seed for reproducibility
            n_channels: Number of channels (6 for basic, 50 for ROSIE-compatible)
        """
        self.image_size = image_size
        self.seed = seed
        self.n_channels = n_channels
        
        if n_channels == 6:
            # Original 6-channel panel
            self.channel_names = ['DAPI', 'CD3', 'CD20', 'CD45', 'CD68', 'CK']
        elif n_channels == 50:
            # Comprehensive 50-channel panel (ROSIE-compatible)
            self.channel_names = self._get_50_channel_panel()
        else:
            # Custom channel count
            self.channel_names = [f'Marker_{i:02d}' for i in range(n_channels)]
            
        self._setup_marker_properties()
    
    def _get_50_channel_panel(self):
        """Return comprehensive 50-marker panel similar to ROSIE capabilities."""
        return [
            # Nuclear and structural (5)
            'DAPI', 'Hoechst', 'PI', 'PCNA', 'Ki67',
            
            # T cell markers (8) 
            'CD3', 'CD4', 'CD8', 'FOXP3', 'CD25', 'PD1', 'CTLA4', 'CD45RO',
            
            # B cell markers (4)
            'CD20', 'CD19', 'CD79a', 'BCL6',
            
            # Myeloid/Macrophage markers (6)
            'CD68', 'CD163', 'CD11b', 'CD11c', 'CD14', 'iNOS',
            
            # Epithelial markers (6)
            'CK', 'EpCAM', 'E-Cadherin', 'CK7', 'CK19', 'Beta-Catenin',
            
            # Immune checkpoints (4)
            'PDL1', 'LAG3', 'TIM3', 'TIGIT',
            
            # Angiogenesis/Endothelial (4)
            'CD31', 'VEGF', 'CD34', 'vWF',
            
            # Stromal/Fibroblast (5)
            'Vimentin', 'SMA', 'FAP', 'Collagen_I', 'Collagen_IV',
            
            # Proliferation/Apoptosis (4)
            'Caspase3', 'PARP', 'p53', 'MDM2',
            
            # Additional functional markers (4)
            'HER2', 'EGFR', 'c-MYC', 'BCL2'
        ]
    
    def _setup_marker_properties(self):
        """Setup biological properties for each marker type."""
        self.marker_properties = {}
        
        for i, marker in enumerate(self.channel_names):
            # Default properties
            props = {
                'base_intensity': 0.3,
                'positive_fraction': 0.1,
                'spatial_clustering': False,
                'cell_type': 'general'
            }
            
            # Specific properties based on marker biology
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
        """
        Create base cellular structure with nuclei.
        
        Args:
            cell_density: Proportion of image covered by cells (0-1)
            seed_offset: Offset for random seed
        """
        np.random.seed(self.seed + seed_offset)
        
        # Create random cell centers
        n_cells = int(cell_density * self.image_size ** 2 / 100)
        cell_centers_x = np.random.randint(0, self.image_size, n_cells)
        cell_centers_y = np.random.randint(0, self.image_size, n_cells)
        
        # Create nuclear channel (DAPI)
        dapi = np.zeros((self.image_size, self.image_size))
        
        for cx, cy in zip(cell_centers_x, cell_centers_y):
            # Add gaussian blob for nucleus
            y, x = np.ogrid[0:self.image_size, 0:self.image_size]
            nucleus_size = np.random.uniform(3, 6)
            nucleus_intensity = np.random.uniform(0.6, 1.0)
            
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            nucleus = nucleus_intensity * np.exp(-dist**2 / (2 * nucleus_size**2))
            dapi += nucleus
        
        # Smooth and add noise
        dapi = gaussian_filter(dapi, sigma=1.0)
        dapi += np.random.normal(0, 0.02, dapi.shape)
        dapi = np.clip(dapi, 0, 1)
        
        return dapi, (cell_centers_x, cell_centers_y)
    
    def _create_marker_channel(self, cell_centers, base_intensity, 
                               positive_fraction, spatial_clustering=False):
        """
        Create a marker channel (CD3, CD20, etc) with specific expression pattern.
        
        Args:
            cell_centers: Tuple of (x_coords, y_coords) for cells
            base_intensity: Base fluorescence intensity
            positive_fraction: Fraction of cells expressing this marker
            spatial_clustering: Whether positive cells cluster together
        """
        cell_centers_x, cell_centers_y = cell_centers
        n_cells = len(cell_centers_x)
        
        # Determine which cells are positive
        if spatial_clustering:
            # Create hotspots where marker is expressed
            n_hotspots = np.random.randint(1, 4)
            hotspot_centers = np.random.randint(0, self.image_size, (n_hotspots, 2))
            
            # Cells near hotspots are more likely to be positive
            distances = []
            for cx, cy in zip(cell_centers_x, cell_centers_y):
                min_dist = min(np.sqrt((cx - hx)**2 + (cy - hy)**2) 
                              for hx, hy in hotspot_centers)
                distances.append(min_dist)
            
            # Probability decreases with distance
            distances = np.array(distances)
            probs = np.exp(-distances / 50) 
            probs = probs / probs.max() * positive_fraction * 2
            positive_mask = np.random.random(n_cells) < probs
        else:
            # Random distribution
            positive_mask = np.random.random(n_cells) < positive_fraction
        
        # Create channel
        channel = np.zeros((self.image_size, self.image_size))
        
        for i, (cx, cy) in enumerate(zip(cell_centers_x, cell_centers_y)):
            if positive_mask[i]:
                y, x = np.ogrid[0:self.image_size, 0:self.image_size]
                cell_size = np.random.uniform(4, 8)
                intensity = base_intensity * np.random.uniform(0.7, 1.3)
                
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                blob = intensity * np.exp(-dist**2 / (2 * cell_size**2))
                channel += blob
        
        # Smooth and add noise
        channel = gaussian_filter(channel, sigma=1.2)
        channel += np.random.normal(0, 0.01, channel.shape)
        channel = np.clip(channel, 0, 1)
        
        return channel
    
    def generate_tumor_region(self, seed_offset=0):
        """
        Generate IF image for tumor region.
        High epithelial markers, low immune infiltration.
        """
        # High cell density in tumor
        dapi, cell_centers = self._create_cellular_structure(
            cell_density=0.5, seed_offset=seed_offset
        )
        
        channels = []
        
        # Generate all channels based on tissue-specific patterns
        for i, marker in enumerate(self.channel_names):
            if marker in ['DAPI', 'Hoechst', 'PI'] and i == 0:
                # Use the nuclear stain we already created
                channels.append(dapi)
            else:
                props = self.marker_properties[marker].copy()
                
                # Tissue-specific modifications for tumor
                if props['cell_type'] == 'epithelial':
                    props['positive_fraction'] *= 3.0  # High epithelial
                    props['base_intensity'] *= 1.2
                elif props['cell_type'] in ['T_cell', 'B_cell']:
                    props['positive_fraction'] *= 0.3  # Low immune
                    props['spatial_clustering'] = True
                elif props['cell_type'] == 'macrophage':
                    props['positive_fraction'] *= 0.8  # Moderate TAMs
                elif props['cell_type'] == 'proliferation':
                    props['positive_fraction'] *= 2.0  # High proliferation
                elif props['cell_type'] == 'stromal':
                    props['positive_fraction'] *= 0.5  # Reduced stroma
                
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
    
    def generate_immune_aggregate(self, seed_offset=0):
        """
        Generate IF image for immune-rich region.
        High T cells, B cells, macrophages. Low epithelial.
        """
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
                
                # Tissue-specific modifications for immune aggregate
                if props['cell_type'] in ['T_cell', 'B_cell']:
                    props['positive_fraction'] *= 4.0  # Very high immune
                    props['base_intensity'] *= 1.3
                    props['spatial_clustering'] = True
                elif props['cell_type'] == 'macrophage':
                    props['positive_fraction'] *= 2.5  # High macrophages
                    props['spatial_clustering'] = True
                elif props['cell_type'] == 'epithelial':
                    props['positive_fraction'] *= 0.1  # Very low epithelial
                elif props['cell_type'] == 'checkpoint':
                    props['positive_fraction'] *= 3.0  # High checkpoint expression
                elif props['cell_type'] == 'proliferation':
                    props['positive_fraction'] *= 1.5  # Moderate proliferation
                elif props['cell_type'] == 'stromal':
                    props['positive_fraction'] *= 0.3  # Low stroma
                
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
        """
        Generate IF image for stromal region.
        Low cell density, high stromal markers, sparse immune.
        """
        dapi, cell_centers = self._create_cellular_structure(
            cell_density=0.2, seed_offset=seed_offset
        )
        
        channels = []
        
        for i, marker in enumerate(self.channel_names):
            if marker in ['DAPI', 'Hoechst', 'PI'] and i == 0:
                channels.append(dapi)
            else:
                props = self.marker_properties[marker].copy()
                
                # Tissue-specific modifications for stroma
                if props['cell_type'] == 'stromal':
                    props['positive_fraction'] *= 3.0  # High stromal markers
                    props['base_intensity'] *= 1.2
                elif props['cell_type'] in ['T_cell', 'B_cell']:
                    props['positive_fraction'] *= 0.4  # Sparse immune
                elif props['cell_type'] == 'epithelial':
                    props['positive_fraction'] *= 0.05  # Very low epithelial
                elif props['cell_type'] == 'macrophage':
                    props['positive_fraction'] *= 0.6  # Some macrophages
                elif props['cell_type'] == 'endothelial':
                    props['positive_fraction'] *= 1.5  # Some vessels
                elif props['cell_type'] == 'proliferation':
                    props['positive_fraction'] *= 0.2  # Low proliferation
                
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
        """
        Generate IF image for normal tissue.
        Balanced cellular composition with moderate epithelial.
        """
        dapi, cell_centers = self._create_cellular_structure(
            cell_density=0.4, seed_offset=seed_offset
        )
        
        channels = []
        
        for i, marker in enumerate(self.channel_names):
            if marker in ['DAPI', 'Hoechst', 'PI'] and i == 0:
                channels.append(dapi)
            else:
                props = self.marker_properties[marker].copy()
                
                # Tissue-specific modifications for normal tissue
                if props['cell_type'] == 'epithelial':
                    props['positive_fraction'] *= 1.5  # Moderate epithelial
                elif props['cell_type'] in ['T_cell', 'B_cell']:
                    props['positive_fraction'] *= 0.6  # Some immune surveillance
                elif props['cell_type'] == 'macrophage':
                    props['positive_fraction'] *= 0.5  # Resident macrophages
                elif props['cell_type'] == 'stromal':
                    props['positive_fraction'] *= 0.8  # Normal stroma
                elif props['cell_type'] == 'proliferation':
                    props['positive_fraction'] *= 0.3  # Low proliferation
                elif props['cell_type'] == 'checkpoint':
                    props['positive_fraction'] *= 0.2  # Low checkpoint expression
                elif props['cell_type'] == 'endothelial':
                    props['positive_fraction'] *= 0.8  # Normal vasculature
                
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
        """
        Generate IF image based on tissue type annotation.
        
        Args:
            tissue_type: One of ['Tumor', 'Immune_Aggregate', 'Stroma', 'Normal']
            seed_offset: Random seed offset for variability
            
        Returns:
            numpy array of shape (n_channels, height, width)
        """
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
        """
        Generate a batch of IF images for multiple ROIs.
        
        Args:
            tissue_types: List of tissue type strings
            return_tensor: If True, return PyTorch tensor; else numpy
            
        Returns:
            Batch of IF images (batch, channels, height, width)
        """
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
    """Test IF image generation"""
    print("="*60)
    print("Testing Simulated IF Generator")
    print("="*60)
    
    generator = SimulatedIFGenerator(image_size=224, seed=42)
    
    # Test each tissue type
    tissue_types = ['Tumor', 'Immune_Aggregate', 'Stroma', 'Normal']
    
    for tissue in tissue_types:
        print(f"\n{tissue}:")
        img = generator.generate_for_tissue_type(tissue, seed_offset=0)
        print(f"  Shape: {img.shape}")
        print(f"  Channels: {generator.channel_names}")
        print(f"  Intensity ranges:")
        for ch, name in enumerate(generator.channel_names):
            print(f"    {name}: {img[ch].min():.3f} - {img[ch].max():.3f} "
                  f"(mean: {img[ch].mean():.3f})")
    
    # Test batch generation
    print(f"\n{'='*60}")
    print("Testing batch generation...")
    test_types = ['Tumor', 'Tumor', 'Immune_Aggregate', 'Stroma', 'Normal']
    batch = generator.generate_batch(test_types, return_tensor=True)
    print(f"Batch shape: {batch.shape}")
    print(f"Batch dtype: {batch.dtype}")


# Factory functions for different configurations

def create_basic_if_generator(image_size=224, seed=42):
    """Create 6-channel IF generator (current approach)."""
    return SimulatedIFGenerator(image_size=image_size, seed=seed, n_channels=6)


def create_rosie_compatible_if_generator(image_size=224, seed=42):
    """Create 50-channel IF generator (ROSIE-compatible)."""
    return SimulatedIFGenerator(image_size=image_size, seed=seed, n_channels=50)


def create_if_generator(n_channels=50, image_size=224, seed=42):
    """Create IF generator with specified number of channels.
    
    Args:
        n_channels: Number of channels (6 for basic, 50 for ROSIE-compatible)
        image_size: Size of generated images
        seed: Random seed for reproducibility
    """
    return SimulatedIFGenerator(image_size=image_size, seed=seed, n_channels=n_channels)
    print(f"Batch range: {batch.min():.3f} - {batch.max():.3f}")
    
    print(f"\n{'='*60}")
    print("✅ Generator test complete!")
    print("="*60)
    
    return generator


if __name__ == '__main__':
    generator = test_generator()
