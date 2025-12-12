

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
import cv2
from typing import Dict, List, Optional, Tuple, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ROSIEModel:
    
    def __init__(self, model_path: Union[str, Path] = "ROSIE.pth", device: str = "auto"):
        self.model_path = Path(model_path)
        self.device = self._setup_device(device)
        
        self.channel_mapping = {
            'DAPI': 0,
            'CD3': 12,
            'CD20': 23,
            'CD45': 34,
            'CD68': 41,
            'CK': 47
        }
        
        self.channel_names = list(self.channel_mapping.keys())
        
        self.model = None
        self._model_loaded = False
        
        logger.info(f"ROSIEModel initialized with device: {self.device}")
        logger.info(f"Model path: {self.model_path}")
        
    def _setup_device(self, device: str) -> torch.device:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        torch_device = torch.device(device)
        
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, using CPU")
            torch_device = torch.device("cpu")
            
        return torch_device
    
    def _load_model(self) -> nn.Module:
        if not self.model_path.exists():
            raise FileNotFoundError(f"ROSIE model not found: {self.model_path}")
            
        logger.info(f"Loading ROSIE model from {self.model_path}")
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    model = checkpoint['model']
                    logger.info("Loaded model from checkpoint['model']")
                elif 'state_dict' in checkpoint:
                    logger.error("State dict format detected, but model architecture unknown")
                    logger.error("Please provide the complete ROSIE model or architecture details")
                    raise NotImplementedError("State dict loading requires model architecture definition")
                else:
                    model = checkpoint
                    logger.info("Loaded model directly from checkpoint")
            else:
                model = checkpoint
                logger.info("Loaded model directly")
            
            model = model.to(self.device)
            model.eval()
            
            # Log model info
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"ROSIE model loaded successfully: {total_params:,} parameters")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load ROSIE model: {e}")
            raise RuntimeError(f"Could not load ROSIE model from {self.model_path}: {e}")
    
    def _ensure_model_loaded(self):
        """Ensure model is loaded (lazy loading)"""
        if not self._model_loaded:
            self.model = self._load_model()
            self._model_loaded = True
    
    def preprocess_he_image(self, he_image: np.ndarray) -> torch.Tensor:
        """
        Preprocess H&E image for ROSIE input.
        
        Args:
            he_image: H&E image as numpy array (H, W, 3) in RGB format
            
        Returns:
            Preprocessed tensor ready for ROSIE model
        """
        # Ensure RGB format
        if len(he_image.shape) != 3 or he_image.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H, W, 3), got {he_image.shape}")
        
        # Convert to tensor and reorder to (C, H, W)
        he_tensor = torch.from_numpy(he_image).permute(2, 0, 1).float()
        
        # Normalize to [0, 1] if needed
        if he_tensor.max() > 1.0:
            he_tensor = he_tensor / 255.0
        
        # Add batch dimension: (1, C, H, W)
        he_tensor = he_tensor.unsqueeze(0)
        
        # Move to device
        he_tensor = he_tensor.to(self.device)
        
        return he_tensor
    
    def generate_full_if(self, he_image: np.ndarray) -> torch.Tensor:
        """
        Generate full 50-plex IF from H&E image.
        
        Args:
            he_image: H&E image (H, W, 3) RGB
            
        Returns:
            50-channel IF tensor (50, H, W)
        """
        self._ensure_model_loaded()
        
        # Preprocess H&E
        he_tensor = self.preprocess_he_image(he_image)
        
        # Generate IF with ROSIE
        with torch.no_grad():
            full_if = self.model(he_tensor)  # Expected shape: (1, 50, H, W)
        
        # Remove batch dimension and move to CPU
        full_if = full_if.squeeze(0).cpu()  # (50, H, W)
        
        return full_if
    
    def generate_if_6channel(self, he_image: np.ndarray) -> np.ndarray:
        """
        Generate 6-channel IF from H&E image (main function for IF2RNA).
        
        Args:
            he_image: H&E image (H, W, 3) RGB
            
        Returns:
            6-channel IF array (6, H, W) with channels: DAPI, CD3, CD20, CD45, CD68, CK
        """
        # Generate full 50-plex IF
        full_if = self.generate_full_if(he_image)
        
        # Extract our 6 channels
        selected_channels = []
        for channel_name in self.channel_names:
            channel_idx = self.channel_mapping[channel_name]
            
            # Handle potential index errors
            if channel_idx >= full_if.shape[0]:
                logger.warning(f"Channel {channel_name} (index {channel_idx}) not found, using zeros")
                selected_channels.append(torch.zeros(full_if.shape[1], full_if.shape[2]))
            else:
                selected_channels.append(full_if[channel_idx])
        
        # Stack to 6-channel IF
        if_6channel = torch.stack(selected_channels, dim=0)  # (6, H, W)
        
        return if_6channel.numpy()
    
    def get_model_info(self) -> Dict:
        """Get information about loaded model"""
        if not self._model_loaded:
            return {"status": "Model not loaded yet"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "status": "Loaded",
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_path": str(self.model_path),
            "output_channels": self.channel_names
        }


class ROSIEIFGenerator:
    """
    IF generator using ROSIE model instead of simulation.
    Compatible with existing IF2RNA dataset classes.
    """
    
    def __init__(self, 
                 rosie_model_path: Union[str, Path] = "ROSIE.pth",
                 he_data_dir: Optional[Union[str, Path]] = None,
                 image_size: int = 224,
                 fallback_to_simulation: bool = True):
        """
        Initialize ROSIE-based IF generator.
        
        Args:
            rosie_model_path: Path to ROSIE model
            he_data_dir: Directory containing H&E slides
            image_size: Output image size (assumed square)
            fallback_to_simulation: Whether to use simulated IF if H&E not found
        """
        self.rosie = ROSIEModel(rosie_model_path)
        self.he_data_dir = Path(he_data_dir) if he_data_dir else None
        self.image_size = image_size
        self.fallback_to_simulation = fallback_to_simulation
        
        # Channel properties (matching SimulatedIFGenerator interface)
        self.channel_names = self.rosie.channel_names
        self.n_channels = len(self.channel_names)
        
        # Fallback generator for when H&E not available
        if fallback_to_simulation:
            from if2rna.simulated_if_generator import SimulatedIFGenerator
            self.fallback_generator = SimulatedIFGenerator(image_size=image_size, seed=42)
        else:
            self.fallback_generator = None
        
        logger.info(f"ROSIEIFGenerator initialized")
        logger.info(f"H&E data dir: {self.he_data_dir}")
        logger.info(f"Fallback simulation: {fallback_to_simulation}")
    
    def find_he_image_for_roi(self, roi_id: str) -> Optional[Path]:
        """
        Find H&E image file corresponding to ROI ID.
        
        Args:
            roi_id: ROI identifier
            
        Returns:
            Path to H&E image file, or None if not found
        """
        if self.he_data_dir is None or not self.he_data_dir.exists():
            return None
        
        # Common H&E file naming patterns
        possible_files = [
            f"{roi_id}.png",
            f"{roi_id}.jpg", 
            f"{roi_id}.tiff",
            f"{roi_id}_HE.png",
            f"{roi_id}_H&E.png",
            f"ROI_{roi_id}.png",
            f"Sample_{roi_id}.png",
        ]
        
        for filename in possible_files:
            file_path = self.he_data_dir / filename
            if file_path.exists():
                return file_path
        
        return None
    
    def load_he_image(self, he_path: Path) -> np.ndarray:
        """
        Load H&E image from file and resize to target size.
        
        Args:
            he_path: Path to H&E image
            
        Returns:
            H&E image as RGB numpy array (H, W, 3)
        """
        try:
            # Load image
            he_image = cv2.imread(str(he_path))
            
            if he_image is None:
                raise ValueError(f"Could not load image from {he_path}")
            
            # Convert BGR to RGB
            he_image = cv2.cvtColor(he_image, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            if he_image.shape[:2] != (self.image_size, self.image_size):
                he_image = cv2.resize(he_image, (self.image_size, self.image_size))
            
            return he_image
            
        except Exception as e:
            logger.error(f"Failed to load H&E image {he_path}: {e}")
            raise
    
    def generate_for_roi(self, roi_id: str, tissue_type: Optional[str] = None, 
                        seed_offset: int = 0) -> np.ndarray:
        """
        Generate IF image for specific ROI.
        
        Args:
            roi_id: ROI identifier
            tissue_type: Tissue type (used for fallback simulation)
            seed_offset: Random seed offset (used for fallback)
            
        Returns:
            6-channel IF image (6, H, W)
        """
        # Try to find H&E image for this ROI
        he_path = self.find_he_image_for_roi(roi_id)
        
        if he_path is not None:
            try:
                # Load H&E and generate IF with ROSIE
                he_image = self.load_he_image(he_path)
                if_image = self.rosie.generate_if_6channel(he_image)
                
                logger.info(f"Generated ROSIE IF for ROI {roi_id} from {he_path.name}")
                return if_image
                
            except Exception as e:
                logger.error(f"Failed to generate ROSIE IF for ROI {roi_id}: {e}")
                # Fall through to simulation fallback
        
        # Fallback to simulation
        if self.fallback_generator is not None and tissue_type is not None:
            logger.warning(f"No H&E found for ROI {roi_id}, using simulated IF")
            return self.fallback_generator.generate_for_tissue_type(tissue_type, seed_offset=seed_offset)
        else:
            raise RuntimeError(f"No H&E image found for ROI {roi_id} and no fallback available")
    
    def generate_for_tissue_type(self, tissue_type: str, seed_offset: int = 0) -> np.ndarray:
        """
        Compatibility method with SimulatedIFGenerator interface.
        Falls back to simulation since no specific ROI provided.
        """
        if self.fallback_generator is not None:
            return self.fallback_generator.generate_for_tissue_type(tissue_type, seed_offset)
        else:
            raise NotImplementedError("Cannot generate by tissue type without H&E data or fallback simulation")
    
    def generate_batch(self, roi_ids: List[str], tissue_types: Optional[List[str]] = None,
                      return_tensor: bool = True) -> Union[torch.Tensor, np.ndarray]:
        """
        Generate batch of IF images for multiple ROIs.
        
        Args:
            roi_ids: List of ROI identifiers
            tissue_types: List of tissue types (for fallback)
            return_tensor: Whether to return PyTorch tensor
            
        Returns:
            Batch of IF images (batch, channels, height, width)
        """
        if tissue_types is None:
            tissue_types = [None] * len(roi_ids)
        
        images = []
        for i, (roi_id, tissue_type) in enumerate(zip(roi_ids, tissue_types)):
            img = self.generate_for_roi(roi_id, tissue_type, seed_offset=i)
            images.append(img)
        
        batch = np.stack(images, axis=0)
        
        if return_tensor:
            batch = torch.from_numpy(batch).float()
        
        logger.info(f"Generated batch of {len(roi_ids)} IF images")
        return batch


def test_rosie_model(model_path: str = "ROSIE.pth"):
    """Test ROSIE model loading and basic functionality"""
    print("="*60)
    print("Testing ROSIE Model")
    print("="*60)
    
    try:
        # Test model loading
        print("Loading ROSIE...")
        rosie = ROSIEModel(model_path)
        
        # Get model info before loading
        info = rosie.get_model_info()
        print(f"Status: {info['status']}")
        
        # Test with dummy H&E image
        print("\n2. Testing with dummy H&E image...")
        dummy_he = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        print(f"   Input H&E shape: {dummy_he.shape}")
        
        # Generate IF
        if_image = rosie.generate_if_from_he(dummy_he)
        print(f"Output: {if_image.shape}")
        
        # Get model info after loading
        info = rosie.get_model_info()
        print(f"Model info: {info}")
        
        # Test IF generator
        print("Testing generator...")
        generator = ROSIEIFGenerator(model_path, fallback_to_simulation=True)
        
        # Test ROI generation (will use simulation since no H&E dir)
        test_if = generator.generate_for_roi("test_roi", tissue_type="Tumor", seed_offset=0)
        print(f"   Generated IF shape: {test_if.shape}")
        print(f"   Channels: {generator.channel_names}")
        
        print("Test passed")
        
        return rosie, generator
        
    except Exception as e:
        print(f"Test failed: {e}")
        raise


if __name__ == '__main__':
    # Test ROSIE model
    rosie_model, rosie_generator = test_rosie_model()