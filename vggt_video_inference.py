import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Union, Optional, Dict
import argparse
from tqdm import tqdm
import os
import sys
import json
import pickle

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


class PredictionSaver:
    """Save VGGT predictions in multiple formats."""
    
    def __init__(self, output_dir: str):
        """
        Initialize prediction saver.
        
        Args:
            output_dir: Directory to save predictions
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir = self.output_dir / "predictions"
        self.predictions_dir.mkdir(exist_ok=True)
    
    def save_predictions(
        self,
        predictions: Dict,
        image_names: List[str],
        formats: List[str] = ["numpy", "pickle", "json_metadata"]
    ) -> Dict[str, Path]:
        """
        Save predictions in multiple formats.
        
        Args:
            predictions: Dictionary of model predictions
            image_names: List of image paths used for inference
            formats: List of formats to save ("numpy", "pickle", "json_metadata", "npy")
            
        Returns:
            Dictionary of saved file paths
        """
        print("\n" + "=" * 60)
        print("SAVING PREDICTIONS")
        print("=" * 60)
        print(f"Predictions directory: {self.predictions_dir}")
        
        saved_paths = {}
        
        # Save as NumPy files
        if "numpy" in formats or "npy" in formats:
            saved_paths.update(self._save_as_numpy(predictions))
        
        # Save as Pickle
        if "pickle" in formats:
            saved_paths.update(self._save_as_pickle(predictions))
        
        # Save metadata as JSON
        if "json_metadata" in formats:
            saved_paths.update(self._save_metadata_as_json(predictions, image_names))
        
        # Save complete predictions as pickle (full data)
        if "full_pickle" in formats:
            saved_paths.update(self._save_full_predictions_pickle(predictions))
        
        # Save depth maps as PNG
        if "depth_png" in formats:
            saved_paths.update(self._save_depth_as_png(predictions))
        
        print(f"\n{'=' * 60}")
        print("PREDICTIONS SAVED")
        print(f"{'=' * 60}")
        print(f"Total files saved: {len(saved_paths)}")
        for name, path in saved_paths.items():
            size = self._get_file_size(path)
            print(f"  ✅ {name}: {path.name} ({size})")
        
        return saved_paths
    
    def _save_as_numpy(self, predictions: Dict) -> Dict[str, Path]:
        """Save predictions as separate NumPy files."""
        saved = {}
        
        print("\nSaving as NumPy arrays (.npy)...")
        
        for key, value in predictions.items():
            if torch.is_tensor(value):
                # Convert to numpy and move to CPU
                np_array = value.cpu().numpy()
                
                # Save individual file
                filepath = self.predictions_dir / f"{key}.npy"
                np.save(str(filepath), np_array)
                saved[f"numpy_{key}"] = filepath
                
                print(f"  ✓ {key}: shape {np_array.shape}, dtype {np_array.dtype}")
            
            elif isinstance(value, list):
                # Handle lists (e.g., pose_enc_list)
                print(f"  ⊘ {key}: list (skipped, use pickle for full data)")
        
        return saved
    
    def _save_as_pickle(self, predictions: Dict) -> Dict[str, Path]:
        """Save predictions as Pickle files (tensor-compatible)."""
        print("\nSaving as Pickle (.pkl)...")
        
        saved = {}
        
        for key, value in predictions.items():
            if torch.is_tensor(value):
                filepath = self.predictions_dir / f"{key}.pkl"
                torch.save(value, str(filepath))
                saved[f"pickle_{key}"] = filepath
                print(f"  ✓ {key}: {filepath.name}")
            
            elif isinstance(value, list):
                filepath = self.predictions_dir / f"{key}_list.pkl"
                with open(filepath, 'wb') as f:
                    pickle.dump(value, f)
                saved[f"pickle_{key}_list"] = filepath
                print(f"  ✓ {key} (list): {filepath.name}")
        
        return saved
    
    def _save_metadata_as_json(self, predictions: Dict, image_names: List[str]) -> Dict[str, Path]:
        """Save prediction metadata (shapes, dtypes, image names) as JSON."""
        print("\nSaving metadata as JSON...")
        
        metadata = {
            "num_images": len(image_names),
            "image_names": image_names,
            "predictions": {}
        }
        
        for key, value in predictions.items():
            if torch.is_tensor(value):
                metadata["predictions"][key] = {
                    "type": "tensor",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "filename": f"{key}.npy"
                }
            elif isinstance(value, list):
                metadata["predictions"][key] = {
                    "type": "list",
                    "length": len(value),
                    "filename": f"{key}_list.pkl"
                }
        
        filepath = self.predictions_dir / "metadata.json"
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Metadata: {filepath.name}")
        
        return {"json_metadata": filepath}
    
    def _save_full_predictions_pickle(self, predictions: Dict) -> Dict[str, Path]:
        """Save complete predictions dictionary as single pickle file."""
        print("\nSaving complete predictions as Pickle...")
        
        filepath = self.predictions_dir / "predictions_full.pkl"
        
        # Convert tensors to numpy for easier loading
        predictions_serializable = {}
        for key, value in predictions.items():
            if torch.is_tensor(value):
                predictions_serializable[key] = value.cpu().numpy()
            else:
                predictions_serializable[key] = value
        
        with open(filepath, 'wb') as f:
            pickle.dump(predictions_serializable, f)
        
        size = filepath.stat().st_size / (1024 * 1024)
        print(f"  ✓ Complete predictions: {filepath.name} ({size:.2f} MB)")
        
        return {"full_pickle": filepath}
    
    def _save_depth_as_png(self, predictions: Dict) -> Dict[str, Path]:
        """Save depth maps as PNG (normalized)."""
        if "depth" not in predictions:
            return {}
        
        print("\nSaving depth maps as PNG...")
        
        depth = predictions["depth"].cpu().numpy()  # [B, N, H, W, 1]
        
        saved = {}
        
        for b in range(depth.shape[0]):
            for n in range(depth.shape[1]):
                depth_map = depth[b, n, :, :, 0]
                
                # Normalize to 0-255
                depth_min = depth_map.min()
                depth_max = depth_map.max()
                
                if depth_max > depth_min:
                    depth_normalized = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
                else:
                    depth_normalized = np.zeros_like(depth_map, dtype=np.uint8)
                
                filepath = self.predictions_dir / f"depth_b{b:02d}_n{n:02d}.png"
                cv2.imwrite(str(filepath), depth_normalized)
                saved[f"depth_png_b{b}_n{n}"] = filepath
        
        print(f"  ✓ Saved {len(saved)} depth maps")
        
        return saved
    
    @staticmethod
    def _get_file_size(filepath: Path) -> str:
        """Get human-readable file size."""
        size_bytes = filepath.stat().st_size
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        
        return f"{size_bytes:.1f} TB"


class VideoFrameExtractor:
    """Extract frames from video for VGGT reconstruction."""
    
    def __init__(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        frame_interval: int = 1,
        max_frames: Optional[int] = None,
        skip_frames: int = 0,
        quality: int = 95,
        resize: Optional[tuple] = None,
        detect_motion_blur: bool = False,
        motion_blur_threshold: float = 100.0
    ):
        """
        Initialize video frame extractor.
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save extracted frames (default: video_name_frames/ in writable location)
            frame_interval: Extract every N-th frame (1 = all frames)
            max_frames: Maximum number of frames to extract
            skip_frames: Number of initial frames to skip
            quality: JPEG quality for saved frames (0-100)
            resize: Optional (width, height) to resize frames
            detect_motion_blur: Skip blurry frames
            motion_blur_threshold: Laplacian variance threshold for blur detection
        """
        self.video_path = Path(video_path)
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Handle output directory - use Kaggle working directory if input is read-only
        if output_dir is None:
            proposed_dir = self.video_path.parent / f"{self.video_path.stem}_frames"
            
            # Check if parent directory is writable
            if self._is_writable(self.video_path.parent):
                self.output_dir = proposed_dir
            else:
                # Fall back to Kaggle working directory or current working directory
                if self._is_kaggle_environment():
                    self.output_dir = Path("/kaggle/working") / f"{self.video_path.stem}_frames"
                else:
                    self.output_dir = Path.cwd() / f"{self.video_path.stem}_frames"
        else:
            self.output_dir = Path(output_dir)
        
        # Ensure output directory is writable
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            # If still can't create in specified location, use fallback
            print(f"Warning: Cannot create directory at {self.output_dir}: {e}")
            if self._is_kaggle_environment():
                self.output_dir = Path("/kaggle/working") / f"{self.video_path.stem}_frames"
            else:
                self.output_dir = Path.cwd() / f"{self.video_path.stem}_frames"
            print(f"Using fallback directory: {self.output_dir}")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.frame_interval = frame_interval
        self.max_frames = max_frames
        self.skip_frames = skip_frames
        self.quality = quality
        self.resize = resize
        self.detect_motion_blur = detect_motion_blur
        self.motion_blur_threshold = motion_blur_threshold
        
        self.extracted_frames: List[str] = []
    
    @staticmethod
    def _is_kaggle_environment() -> bool:
        """Check if running in Kaggle environment."""
        return os.path.exists("/kaggle/working")
    
    @staticmethod
    def _is_writable(path: Path) -> bool:
        """Check if a directory is writable."""
        try:
            test_file = path / ".write_test"
            test_file.touch()
            test_file.unlink()
            return True
        except (OSError, PermissionError):
            return False
    
    def compute_blur_score(self, frame: np.ndarray) -> float:
        """
        Compute Laplacian variance to detect motion blur.
        Higher values = sharper image.
        
        Args:
            frame: Input image (BGR)
            
        Returns:
            Blur score (variance of Laplacian)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return variance
    
    def extract_frames(self) -> List[str]:
        """
        Extract frames from video.
        
        Returns:
            List of paths to extracted frame images
        """
        print("=" * 60)
        print("EXTRACTING FRAMES FROM VIDEO")
        print("=" * 60)
        print(f"Video: {self.video_path}")
        print(f"Output directory: {self.output_dir}")
        
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\nVideo Properties:")
        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Duration: {total_frames/fps:.2f} seconds")
        
        print(f"\nExtraction Settings:")
        print(f"  Frame interval: {self.frame_interval}")
        print(f"  Skip initial frames: {self.skip_frames}")
        print(f"  Max frames: {self.max_frames if self.max_frames else 'unlimited'}")
        print(f"  Motion blur detection: {self.detect_motion_blur}")
        if self.resize:
            print(f"  Resize to: {self.resize[0]}x{self.resize[1]}")
        
        frame_count = 0
        extracted_count = 0
        skipped_blur = 0
        
        with tqdm(total=min(total_frames, self.max_frames) if self.max_frames else total_frames) as pbar:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Skip initial frames
                if frame_count < self.skip_frames:
                    frame_count += 1
                    continue
                
                # Check frame interval
                if (frame_count - self.skip_frames) % self.frame_interval != 0:
                    frame_count += 1
                    pbar.update(1)
                    continue
                
                # Check max frames
                if self.max_frames and extracted_count >= self.max_frames:
                    break
                
                # Motion blur detection
                if self.detect_motion_blur:
                    blur_score = self.compute_blur_score(frame)
                    if blur_score < self.motion_blur_threshold:
                        skipped_blur += 1
                        frame_count += 1
                        pbar.update(1)
                        continue
                
                # Resize if needed
                if self.resize:
                    frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_LANCZOS4)
                
                # Save frame
                frame_filename = f"frame_{extracted_count:06d}.jpg"
                frame_path = self.output_dir / frame_filename
                
                cv2.imwrite(
                    str(frame_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.quality]
                )
                
                self.extracted_frames.append(str(frame_path))
                extracted_count += 1
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        
        print(f"\n{'=' * 60}")
        print(f"EXTRACTION COMPLETE")
        print(f"{'=' * 60}")
        print(f"Total frames processed: {frame_count}")
        print(f"Frames extracted: {extracted_count}")
        if self.detect_motion_blur:
            print(f"Frames skipped (motion blur): {skipped_blur}")
        print(f"Output directory: {self.output_dir}")
        
        return self.extracted_frames
    
    def select_frames_by_parallax(
        self,
        num_frames: int,
        method: str = "uniform"
    ) -> List[str]:
        """
        Select subset of frames with good parallax/baseline.
        
        Args:
            num_frames: Number of frames to select
            method: Selection method ("uniform", "keyframe")
            
        Returns:
            List of selected frame paths
        """
        if not self.extracted_frames:
            raise RuntimeError("No frames extracted yet. Call extract_frames() first.")
        
        if num_frames >= len(self.extracted_frames):
            return self.extracted_frames
        
        if method == "uniform":
            # Uniform sampling
            indices = np.linspace(0, len(self.extracted_frames) - 1, num_frames, dtype=int)
            selected = [self.extracted_frames[i] for i in indices]
        elif method == "keyframe":
            # Simple keyframe selection based on image difference
            selected = self._select_keyframes(num_frames)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        print(f"\nSelected {len(selected)} frames with {method} sampling")
        return selected
    
    def _select_keyframes(self, num_frames: int) -> List[str]:
        """Select keyframes based on image differences."""
        selected = [self.extracted_frames[0]]  # Always include first frame
        
        # Compute differences between consecutive frames
        differences = []
        prev_frame = cv2.imread(self.extracted_frames[0], cv2.IMREAD_GRAYSCALE)
        
        for frame_path in tqdm(self.extracted_frames[1:], desc="Computing frame differences"):
            curr_frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            diff = cv2.absdiff(prev_frame, curr_frame)
            differences.append((frame_path, diff.mean()))
            prev_frame = curr_frame
        
        # Sort by difference and select top frames
        differences.sort(key=lambda x: x[1], reverse=True)
        selected.extend([d[0] for d in differences[:num_frames-2]])
        selected.append(self.extracted_frames[-1])  # Always include last frame
        
        # Sort by original order
        selected.sort(key=lambda x: self.extracted_frames.index(x))
        
        return selected


def run_vggt_inference(
    image_names: List[str],
    model_name: str = "facebook/VGGT-1B",
    device: Optional[str] = None,
    batch_size: int = 1,
    save_predictions: bool = True,
    predictions_output_dir: Optional[str] = None
) -> Dict:
    """
    Run VGGT inference on images.
    
    Args:
        image_names: List of image paths
        model_name: VGGT model name
        device: Device to run on (auto-detect if None)
        batch_size: Process images in batches
        save_predictions: Whether to save predictions
        predictions_output_dir: Directory to save predictions
        
    Returns:
        Dictionary of predictions
    """
    print("\n" + "=" * 60)
    print("RUNNING VGGT INFERENCE")
    print("=" * 60)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {device}")
    
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        print(f"Precision: {dtype}")
    else:
        dtype = torch.float32
    
    # Initialize the model and load the pretrained weights
    print(f"\nLoading model: {model_name}")
    print("(This will download weights on first run)")
    model = VGGT.from_pretrained(model_name).to(device)
    
    print(f"\nProcessing {len(image_names)} images...")
    
    # Load and preprocess images
    images = load_and_preprocess_images(image_names).to(device)
    
    print(f"Image tensor shape: {images.shape}")
    
    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        if device == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        else:
            predictions = model(images)
    
    print("Inference complete!")
    
    # Print prediction info
    print("\nPredictions:")
    for key, value in predictions.items():
        if torch.is_tensor(value):
            print(f"  {key}: {value.shape} ({value.dtype})")
        else:
            print(f"  {key}: {type(value)}")
    
    # Save predictions
    if save_predictions:
        if predictions_output_dir is None:
            if os.path.exists("/kaggle/working"):
                predictions_output_dir = "/kaggle/working"
            else:
                predictions_output_dir = os.getcwd()
        
        saver = PredictionSaver(predictions_output_dir)
        saved_paths = saver.save_predictions(
            predictions=predictions,
            image_names=image_names,
            formats=["numpy", "pickle", "json_metadata", "full_pickle", "depth_png"]
        )
        
        predictions["_saved_paths"] = saved_paths
    
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Extract video frames and run VGGT reconstruction"
    )
    
    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video",
        type=str,
        help="Path to input video file"
    )
    input_group.add_argument(
        "--images",
        type=str,
        nargs="+",
        help="List of image paths (skip video extraction)"
    )
    input_group.add_argument(
        "--image_dir",
        type=str,
        help="Directory containing images (skip video extraction)"
    )
    
    # Frame extraction
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save extracted frames (defaults to /kaggle/working/ on Kaggle)"
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=10,
        help="Extract every N-th frame (default: 10)"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=100,
        help="Maximum number of frames to extract (default: 100)"
    )
    parser.add_argument(
        "--skip_frames",
        type=int,
        default=0,
        help="Skip initial N frames (default: 0)"
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality for saved frames (default: 95)"
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Resize frames to specified dimensions"
    )
    parser.add_argument(
        "--detect_blur",
        action="store_true",
        help="Skip frames with motion blur"
    )
    parser.add_argument(
        "--blur_threshold",
        type=float,
        default=100.0,
        help="Motion blur threshold (default: 100.0)"
    )
    
    # Frame selection
    parser.add_argument(
        "--select_frames",
        type=int,
        help="Select N frames with good parallax"
    )
    parser.add_argument(
        "--selection_method",
        type=str,
        default="uniform",
        choices=["uniform", "keyframe"],
        help="Frame selection method (default: uniform)"
    )
    
    # VGGT inference
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/VGGT-1B",
        help="VGGT model name (default: facebook/VGGT-1B)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Device to run inference on (auto-detect if not specified)"
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Only extract frames, skip VGGT inference"
    )
    
    # Predictions saving
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        default=True,
        help="Save predictions to disk (default: True)"
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        help="Directory to save predictions (default: /kaggle/working/)"
    )
    parser.add_argument(
        "--skip_save_predictions",
        action="store_true",
        help="Don't save predictions"
    )
    
    args = parser.parse_args()
    
    # Get image names
    if args.video:
        # Extract frames from video
        extractor = VideoFrameExtractor(
            video_path=args.video,
            output_dir=args.output_dir,
            frame_interval=args.frame_interval,
            max_frames=args.max_frames,
            skip_frames=args.skip_frames,
            quality=args.quality,
            resize=tuple(args.resize) if args.resize else None,
            detect_motion_blur=args.detect_blur,
            motion_blur_threshold=args.blur_threshold
        )
        
        image_names = extractor.extract_frames()
        
        # Select subset if requested
        if args.select_frames:
            image_names = extractor.select_frames_by_parallax(
                num_frames=args.select_frames,
                method=args.selection_method
            )
    
    elif args.images:
        image_names = args.images
    
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_names = sorted([
            str(p) for p in image_dir.iterdir()
            if p.suffix.lower() in image_extensions
        ])
        print(f"Found {len(image_names)} images in {image_dir}")
    
    # Run VGGT inference
    if not args.skip_inference:
        if not image_names:
            print("Error: No images to process!")
            return
        
        save_preds = args.save_predictions and not args.skip_save_predictions
        
        predictions = run_vggt_inference(
            image_names=image_names,
            model_name=args.model,
            device=args.device,
            save_predictions=save_preds,
            predictions_output_dir=args.predictions_dir
        )
        
        # Print saved paths
        if save_preds and "_saved_paths" in predictions:
            print("\n" + "=" * 60)
            print("SAVED FILES SUMMARY")
            print("=" * 60)
            for name, path in predictions["_saved_paths"].items():
                print(f"✅ {name}")
                print(f"   Path: {path}")
    
    else:
        print(f"\nSkipping inference. {len(image_names)} frames ready for processing.")
        print("\nImage names:")
        for i, name in enumerate(image_names[:10]):  # Show first 10
            print(f"  {i+1}. {name}")
        if len(image_names) > 10:
            print(f"  ... and {len(image_names) - 10} more")


if __name__ == "__main__":
    main()
