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
    
    Channels (standard GeoMx panel):
    0. DAPI - nuclear stain (all cells)
    1. CD3 - T cells
    2. CD20 - B cells  
    3. CD45 - pan-leukocyte
    4. CD68 - macrophages
    5. CK (Pan-Cytokeratin) - epithelial cells
    """
    
    def __init__(self, image_size=224, seed=42):
        """
        Args:
            image_size: Size of generated square images
            seed: Random seed for reproducibility
        """
        self.image_size = image_size
        self.seed = seed
        self.channel_names = ['DAPI', 'CD3', 'CD20', 'CD45', 'CD68', 'CK']
        self.n_channels = len(self.channel_names)
        
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
        High epithelial (CK), low immune markers.
        """
        # High cell density in tumor
        dapi, cell_centers = self._create_cellular_structure(
            cell_density=0.5, seed_offset=seed_offset
        )
        
        channels = [dapi]
        
        # CD3 (T cells) - sparse in tumor
        channels.append(self._create_marker_channel(
            cell_centers, base_intensity=0.3, positive_fraction=0.05,
            spatial_clustering=True
        ))
        
        # CD20 (B cells) - very sparse
        channels.append(self._create_marker_channel(
            cell_centers, base_intensity=0.2, positive_fraction=0.02,
            spatial_clustering=True
        ))
        
        # CD45 (leukocytes) - sparse
        channels.append(self._create_marker_channel(
            cell_centers, base_intensity=0.4, positive_fraction=0.08,
            spatial_clustering=True
        ))
        
        # CD68 (macrophages) - moderate
        channels.append(self._create_marker_channel(
            cell_centers, base_intensity=0.4, positive_fraction=0.1,
            spatial_clustering=True
        ))
        
        # CK (epithelial) - HIGH in tumor!
        channels.append(self._create_marker_channel(
            cell_centers, base_intensity=0.8, positive_fraction=0.7,
            spatial_clustering=False
        ))
        
        return np.stack(channels, axis=0)
    
    def generate_immune_aggregate(self, seed_offset=0):
        """
        Generate IF image for immune-rich region.
        High CD3, CD45, moderate CD20 and CD68.
        """
        dapi, cell_centers = self._create_cellular_structure(
            cell_density=0.6, seed_offset=seed_offset
        )
        
        channels = [dapi]
        
        # CD3 - HIGH (lots of T cells)
        channels.append(self._create_marker_channel(
            cell_centers, base_intensity=0.8, positive_fraction=0.5,
            spatial_clustering=True
        ))
        
        # CD20 - moderate (some B cells)
        channels.append(self._create_marker_channel(
            cell_centers, base_intensity=0.6, positive_fraction=0.2,
            spatial_clustering=True
        ))
        
        # CD45 - HIGH (pan leukocyte)
        channels.append(self._create_marker_channel(
            cell_centers, base_intensity=0.9, positive_fraction=0.6,
            spatial_clustering=False
        ))
        
        # CD68 - moderate (some macrophages)
        channels.append(self._create_marker_channel(
            cell_centers, base_intensity=0.5, positive_fraction=0.15,
            spatial_clustering=True
        ))
        
        # CK - very low (no epithelial cells)
        channels.append(self._create_marker_channel(
            cell_centers, base_intensity=0.1, positive_fraction=0.05,
            spatial_clustering=False
        ))
        
        return np.stack(channels, axis=0)
    
    def generate_stroma_region(self, seed_offset=0):
        """
        Generate IF image for stromal region.
        Low cell density, sparse immune cells, no epithelial.
        """
        dapi, cell_centers = self._create_cellular_structure(
            cell_density=0.2, seed_offset=seed_offset
        )
        
        channels = [dapi]
        
        # All immune markers low to moderate
        channels.append(self._create_marker_channel(
            cell_centers, base_intensity=0.3, positive_fraction=0.15,
            spatial_clustering=True
        ))  # CD3
        
        channels.append(self._create_marker_channel(
            cell_centers, base_intensity=0.2, positive_fraction=0.08,
            spatial_clustering=True
        ))  # CD20
        
        channels.append(self._create_marker_channel(
            cell_centers, base_intensity=0.4, positive_fraction=0.2,
            spatial_clustering=True
        ))  # CD45
        
        channels.append(self._create_marker_channel(
            cell_centers, base_intensity=0.5, positive_fraction=0.2,
            spatial_clustering=False
        ))  # CD68 (some macrophages)
        
        channels.append(self._create_marker_channel(
            cell_centers, base_intensity=0.1, positive_fraction=0.02,
            spatial_clustering=False
        ))  # CK (very low)
        
        return np.stack(channels, axis=0)
    
    def generate_normal_region(self, seed_offset=0):
        """
        Generate IF image for normal tissue.
        Moderate epithelial, sparse immune.
        """
        dapi, cell_centers = self._create_cellular_structure(
            cell_density=0.4, seed_offset=seed_offset
        )
        
        channels = [dapi]
        
        # Sparse immune cells
        channels.append(self._create_marker_channel(
            cell_centers, base_intensity=0.2, positive_fraction=0.08,
            spatial_clustering=False
        ))  # CD3
        
        channels.append(self._create_marker_channel(
            cell_centers, base_intensity=0.15, positive_fraction=0.05,
            spatial_clustering=False
        ))  # CD20
        
        channels.append(self._create_marker_channel(
            cell_centers, base_intensity=0.3, positive_fraction=0.12,
            spatial_clustering=False
        ))  # CD45
        
        channels.append(self._create_marker_channel(
            cell_centers, base_intensity=0.3, positive_fraction=0.08,
            spatial_clustering=False
        ))  # CD68
        
        # Moderate epithelial (normal tissue structure)
        channels.append(self._create_marker_channel(
            cell_centers, base_intensity=0.5, positive_fraction=0.4,
            spatial_clustering=False
        ))  # CK
        
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
    print(f"Batch range: {batch.min():.3f} - {batch.max():.3f}")
    
    print(f"\n{'='*60}")
    print("✅ Generator test complete!")
    print("="*60)
    
    return generator


if __name__ == '__main__':
    generator = test_generator()
