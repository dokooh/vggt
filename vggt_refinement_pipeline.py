"""
VGGT Point Cloud Refinement Pipeline
=====================================
Improves measurement accuracy of VGGT reconstruction outputs through:
1. Bundle Adjustment (BA)
2. Metric scaling with Ground Control Points
3. Depth map unprojection
4. Point cloud post-processing
"""

import numpy as np
import open3d as o3d
import argparse
import subprocess
import json
import pickle
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import struct
import cv2
import sys
import os
from tqdm import tqdm
import gc
import torch


# Check for optional dependencies
OPTIONAL_DEPS = {}

try:
    import trimesh
    OPTIONAL_DEPS['trimesh'] = True
except ImportError:
    OPTIONAL_DEPS['trimesh'] = False
    print("⚠️  trimesh not installed - 3D visualization features disabled")

try:
    import pycolmap
    OPTIONAL_DEPS['pycolmap'] = True
except ImportError:
    OPTIONAL_DEPS['pycolmap'] = False
    print("⚠️  pycolmap not installed - COLMAP integration features disabled")


class VGGTRefinementPipeline:
    """Pipeline for refining VGGT point cloud reconstructions."""
    
    def __init__(self, scene_dir: str, output_dir: Optional[str] = None):
        """
        Initialize the refinement pipeline.
        
        Args:
            scene_dir: Path to VGGT scene directory
            output_dir: Output directory for refined results (default: scene_dir/refined)
        """
        self.scene_dir = Path(scene_dir)
        self.output_dir = Path(output_dir) if output_dir else self.scene_dir / "refined"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.colmap_dir = self.output_dir / "colmap_ba"
        self.colmap_dir.mkdir(exist_ok=True)
        
        self.depths_dir = self.output_dir / "depths"
        self.depths_dir.mkdir(exist_ok=True)
        
        self.pointclouds_dir = self.output_dir / "pointclouds"
        self.pointclouds_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print("VGGT REFINEMENT PIPELINE")
        print(f"{'='*60}")
        print(f"Scene directory: {self.scene_dir}")
        print(f"Output directory: {self.output_dir}")
    
    # ==================== STEP 1: Bundle Adjustment ====================
    
    def check_vggt_demo_script(self) -> bool:
        """Check if demo_colmap.py exists in VGGT installation."""
        possible_paths = [
            Path.cwd() / "demo_colmap.py",
            self.scene_dir / "demo_colmap.py",
            Path("/kaggle/working/vggt/demo_colmap.py"),
            Path.home() / "vggt" / "demo_colmap.py",
        ]
        
        for path in possible_paths:
            if path.exists():
                print(f"Found demo_colmap.py at: {path}")
                return True
        
        print("⚠️  demo_colmap.py not found. Make sure VGGT is properly installed.")
        return False
    
    def check_vggt_module(self) -> bool:
        """Check if vggt module is installed."""
        try:
            import vggt
            return True
        except ImportError:
            return False
    
    def run_bundle_adjustment(
        self, 
        max_query_pts: int = 4096,
        query_frame_num: int = 3,
        use_ba: bool = True,
        demo_script_path: Optional[str] = None,
        reduce_memory: bool = False,
        disable_fine_tracking: bool = False,
        memory_fraction: Optional[float] = None
    ) -> Optional[Path]:
        """
        Run Bundle Adjustment using VGGT's demo_colmap.py script.
        
        Args:
            max_query_pts: Maximum query points for BA
            query_frame_num: Number of query frames
            use_ba: Enable bundle adjustment
            demo_script_path: Path to demo_colmap.py (auto-detect if None)
            reduce_memory: Reduce memory usage by lowering query points and frames
            disable_fine_tracking: Disable fine tracking (saves significant memory)
            memory_fraction: Limit GPU memory usage (0.0-1.0, e.g., 0.5 for 50%)
            
        Returns:
            Path to COLMAP output directory, or None if failed
        """
        print("=" * 60)
        print("STEP 1: Running Bundle Adjustment")
        print("=" * 60)
        
        # Reduce memory parameters if requested
        if reduce_memory:
            print("\n⚠️  Memory reduction mode enabled")
            max_query_pts = min(max_query_pts, 512)
            query_frame_num = 1
            disable_fine_tracking = True
            print(f"   Reduced query points to: {max_query_pts}")
            print(f"   Reduced query frames to: {query_frame_num}")
            print(f"   Disabled fine tracking to save memory")
        
        # Setup GPU memory management
        gpu_env = os.environ.copy()
        if memory_fraction:
            print(f"\n🔧 Setting GPU memory limit to {memory_fraction*100:.0f}%")
            gpu_env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            # Note: Setting fraction requires specific PyTorch config
            print("   Using expandable_segments for better memory management")
        else:
            # Default: enable expandable segments to avoid fragmentation
            gpu_env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            gpu_env['CUDA_LAUNCH_BLOCKING'] = '1'
        
        # Find demo_colmap.py
        if demo_script_path is None:
            if not self.check_vggt_demo_script():
                print("\n⚠️  SKIPPING Bundle Adjustment - demo_colmap.py not found")
                print("Make sure VGGT is properly installed:")
                print("  git clone https://github.com/facebookresearch/vggt.git")
                print("  cd vggt && pip install -e .")
                return None
            
            # Try to find it
            demo_script_path = "demo_colmap.py"
        
        # Construct command for VGGT's BA script
        cmd = [
            sys.executable, str(demo_script_path),
            f"--scene_dir={self.scene_dir}",
            f"--max_query_pts={max_query_pts}",
            f"--query_frame_num={query_frame_num}",
        ]
        
        if use_ba:
            cmd.append("--use_ba")
        
        if disable_fine_tracking:
            cmd.append("--no_fine_tracking")
            print("\n⚠️  Fine tracking disabled (for memory efficiency)")
        
        # Check if vggt module is installed
        if not self.check_vggt_module():
            print("\n❌ Error: vggt module not installed!")
            print("\nThe VGGT package needs to be installed. Run:")
            print("  cd /kaggle/working/vggt")
            print("  pip install -e .")
            print("\nOr install from the repository:")
            print("  pip install git+https://github.com/facebookresearch/vggt.git")
            return None
        
        print(f"Running command: {' '.join(cmd)}")
        print("\nNote: This may take a while (especially on first run)...\n")
        
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                cwd=self.scene_dir,
                env=gpu_env
            )
            print("✅ Bundle Adjustment completed successfully!")
            print(result.stdout)
            return self.colmap_dir
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running Bundle Adjustment:")
            print(f"Return code: {e.returncode}")
            print(f"\nSTDOUT:\n{e.stdout}")
            print(f"\nSTDERR:\n{e.stderr}")
            
            # Check for reconstruction failure
            if "No reconstruction can be built with BA" in e.stderr or "No reconstruction can be built" in e.stdout:
                print(f"\n🔴 BUNDLE ADJUSTMENT FAILED - No Reconstruction Built")
                print("\nThis means the frames don't have enough feature matches for BA.")
                print("\nRecommendations:")
                print("  1. Use the VGGT point cloud directly (skip BA):")
                print("     python vggt_refinement_pipeline.py --point_cloud output.pkl --no_denoise")
                print("  2. Reduce matching threshold (edit demo_colmap.py):")
                print("     Increase 'vis_thresh' or reduce 'conf_thres_value'")
                print("  3. Use more frames in the video")
                return None
            
            # Check for CUDA out of memory
            if "OutOfMemoryError" in e.stderr or "out of memory" in e.stderr.lower():
                print(f"\n🔴 CUDA OUT OF MEMORY ERROR")
                print("\nSuggested fixes (try in order):")
                print("  1. BEST: Skip BA and just use point cloud refinement:")
                print("     python vggt_refinement_pipeline.py --point_cloud output.pkl --no_denoise")
                print("  2. Disable fine tracking (saves ~50% memory):")
                print("     python vggt_refinement_pipeline.py --use_ba --disable_fine_tracking")
                print("  3. Ultra memory reduction:")
                print("     python vggt_refinement_pipeline.py --use_ba --reduce_memory")
                print("  4. Very aggressive settings:")
                print("     python vggt_refinement_pipeline.py --use_ba --max_query_pts 256 --query_frame_num 1 --disable_fine_tracking")
                print("  5. Run on CPU (very slow):")
                print("     CUDA_VISIBLE_DEVICES='' python vggt_refinement_pipeline.py --use_ba ...")
                return None
            
            # Check for missing dependencies
            if "ModuleNotFoundError" in e.stderr or "ImportError" in e.stderr:
                missing_module = None
                for line in e.stderr.split('\n'):
                    if "ModuleNotFoundError" in line or "ImportError" in line:
                        missing_module = line
                        break
                
                print(f"\n⚠️  Missing dependency detected: {missing_module}")
                print("\nInstall missing dependencies with:")
                print("  pip install trimesh pycolmap")
            
            return None
    
    # ==================== STEP 2: Metric Scaling with GCPs ====================
    
    def load_ground_control_points(self, gcp_file: str) -> List[Dict]:
        """
        Load Ground Control Points from JSON file.
        
        Expected format:
        [
            {
                "image_name": "frame_001.jpg",
                "pixel_coords": [x, y],
                "world_coords": [X, Y, Z],  # in meters
                "description": "marker_1"
            },
            ...
        ]
        """
        with open(gcp_file, 'r') as f:
            gcps = json.load(f)
        print(f"Loaded {len(gcps)} Ground Control Points from {gcp_file}")
        return gcps
    
    def apply_metric_scaling(
        self, 
        point_cloud: o3d.geometry.PointCloud,
        gcp_file: Optional[str] = None,
        known_distance: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
    ) -> Tuple[o3d.geometry.PointCloud, Dict]:
        """
        Scale point cloud to metric units using GCPs or known distances.
        
        Args:
            point_cloud: Input point cloud
            gcp_file: Path to GCP JSON file
            known_distance: Tuple of (point1, point2, actual_distance_meters)
            
        Returns:
            Scaled point cloud and transformation parameters
        """
        print("=" * 60)
        print("STEP 2: Applying Metric Scaling")
        print("=" * 60)
        
        if known_distance is not None:
            # Simple scaling based on known distance
            p1, p2, actual_dist = known_distance
            current_dist = np.linalg.norm(p1 - p2)
            scale = actual_dist / current_dist
            
            transform = {
                'scale': scale,
                'rotation': np.eye(3),
                'translation': np.zeros(3)
            }
            
            print(f"✅ Computed scale factor: {scale:.6f}")
            print(f"   Current distance: {current_dist:.3f} → Actual: {actual_dist:.3f}m")
            
        elif gcp_file is not None:
            # Use GCPs for similarity transformation
            gcps = self.load_ground_control_points(gcp_file)
            transform = self._compute_similarity_transform_from_gcps(point_cloud, gcps)
        else:
            print("⚠️  No scaling information provided. Skipping metric scaling.")
            return point_cloud, {'scale': 1.0, 'rotation': np.eye(3), 'translation': np.zeros(3)}
        
        # Apply transformation
        scaled_pc = self._apply_similarity_transform(point_cloud, transform)
        
        return scaled_pc, transform
    
    def _compute_similarity_transform_from_gcps(
        self, 
        point_cloud: o3d.geometry.PointCloud, 
        gcps: List[Dict]
    ) -> Dict:
        """Compute 7-DoF similarity transform from GCPs."""
        if len(gcps) < 3:
            raise ValueError("Need at least 3 GCPs for similarity transform")
        
        # Extract point correspondences
        src_points = []
        dst_points = []
        
        for gcp in gcps:
            world_coord = np.array(gcp['world_coords'])
            dst_points.append(world_coord)
            
            if 'approx_3d' in gcp:
                src_points.append(np.array(gcp['approx_3d']))
        
        src_points = np.array(src_points)
        dst_points = np.array(dst_points)
        
        # Compute centroids
        src_center = src_points.mean(axis=0)
        dst_center = dst_points.mean(axis=0)
        
        src_centered = src_points - src_center
        dst_centered = dst_points - dst_center
        
        # Compute scale
        src_scale = np.sqrt((src_centered ** 2).sum() / len(src_points))
        dst_scale = np.sqrt((dst_centered ** 2).sum() / len(dst_points))
        scale = dst_scale / src_scale
        
        # Compute rotation using SVD
        H = src_centered.T @ dst_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = dst_center - scale * R @ src_center
        
        print(f"✅ Similarity transform computed:")
        print(f"   Scale: {scale:.6f}")
        print(f"   Translation: {t}")
        
        return {'scale': scale, 'rotation': R, 'translation': t}
    
    def _apply_similarity_transform(
        self, 
        point_cloud: o3d.geometry.PointCloud, 
        transform: Dict
    ) -> o3d.geometry.PointCloud:
        """Apply 7-DoF similarity transformation to point cloud."""
        points = np.asarray(point_cloud.points)
        
        # Apply: p_new = s * R * p_old + t
        points_transformed = (
            transform['scale'] * (points @ transform['rotation'].T) + 
            transform['translation']
        )
        
        transformed_pc = o3d.geometry.PointCloud()
        transformed_pc.points = o3d.utility.Vector3dVector(points_transformed)
        
        if point_cloud.has_colors():
            transformed_pc.colors = point_cloud.colors
        if point_cloud.has_normals():
            # Rotate normals (scale doesn't affect unit normals)
            normals = np.asarray(point_cloud.normals)
            transformed_pc.normals = o3d.utility.Vector3dVector(
                normals @ transform['rotation'].T
            )
        
        return transformed_pc
    
    # ==================== STEP 3: Depth Map Unprojection ====================
    
    def unproject_depth_map_to_point_map(
        self,
        depth_map: np.ndarray,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        color_image: Optional[np.ndarray] = None
    ) -> o3d.geometry.PointCloud:
        """
        Unproject depth map to 3D point cloud using camera parameters.
        
        Args:
            depth_map: HxW depth map
            extrinsic: 4x4 camera extrinsic matrix (world to camera)
            intrinsic: 3x3 camera intrinsic matrix
            color_image: Optional HxW x3 RGB image
            
        Returns:
            Point cloud in world coordinates
        """
        h, w = depth_map.shape
        
        # Create pixel grid
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        u = u.flatten()
        v = v.flatten()
        depth = depth_map.flatten()
        
        # Filter invalid depths
        valid_mask = depth > 0
        u = u[valid_mask]
        v = v[valid_mask]
        depth = depth[valid_mask]
        
        # Unproject to camera coordinates
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        
        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = depth
        
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
        
        # Transform to world coordinates
        cam_to_world = np.linalg.inv(extrinsic)
        points_world_homogeneous = points_cam @ cam_to_world[:3, :3].T + cam_to_world[:3, 3]
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world_homogeneous)
        
        # Add colors if available
        if color_image is not None:
            colors = color_image.reshape(-1, 3)[valid_mask] / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    
    def fuse_depth_maps(
        self,
        depth_maps: List[np.ndarray],
        extrinsics: List[np.ndarray],
        intrinsics: List[np.ndarray],
        color_images: Optional[List[np.ndarray]] = None
    ) -> o3d.geometry.PointCloud:
        """
        Fuse multiple depth maps into a single point cloud.
        
        Args:
            depth_maps: List of depth maps
            extrinsics: List of 4x4 extrinsic matrices
            intrinsics: List of 3x3 intrinsic matrices
            color_images: Optional list of color images
            
        Returns:
            Fused point cloud
        """
        print("=" * 60)
        print("STEP 3: Fusing Depth Maps")
        print("=" * 60)
        
        all_points = []
        all_colors = []
        
        for i, (depth, ext, intr) in enumerate(zip(depth_maps, extrinsics, intrinsics)):
            color = color_images[i] if color_images else None
            pcd = self.unproject_depth_map_to_point_map(depth, ext, intr, color)
            
            all_points.append(np.asarray(pcd.points))
            if pcd.has_colors():
                all_colors.append(np.asarray(pcd.colors))
            
            print(f"✓ Unprojected depth map {i+1}/{len(depth_maps)}: {len(pcd.points)} points")
        
        # Combine all points
        fused_pcd = o3d.geometry.PointCloud()
        fused_pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
        
        if all_colors:
            fused_pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
        
        print(f"✅ Fused point cloud: {len(fused_pcd.points)} total points")
        
        return fused_pcd
    
    # ==================== STEP 4: Post-Processing ====================
    
    def post_process_point_cloud(
        self,
        point_cloud: o3d.geometry.PointCloud,
        voxel_size: float = 0.01,
        statistical_nb_neighbors: int = 20,
        statistical_std_ratio: float = 2.0,
        radius_filter_nb_points: int = 16,
        radius_filter_radius: float = 0.05,
        remove_statistical_outliers: bool = True,
        remove_radius_outliers: bool = True,
        downsample: bool = True
    ) -> o3d.geometry.PointCloud:
        """
        Apply post-processing filters to clean point cloud.
        
        Args:
            point_cloud: Input point cloud
            voxel_size: Voxel size for downsampling (in meters if scaled)
            statistical_nb_neighbors: Neighbors for statistical outlier removal
            statistical_std_ratio: Std ratio for statistical outlier removal
            radius_filter_nb_points: Min neighbors for radius outlier removal
            radius_filter_radius: Search radius for radius outlier removal
            remove_statistical_outliers: Enable statistical outlier removal
            remove_radius_outliers: Enable radius outlier removal
            downsample: Enable voxel downsampling
            
        Returns:
            Cleaned point cloud
        """
        print("=" * 60)
        print("STEP 4: Post-Processing Point Cloud")
        print("=" * 60)
        
        cleaned_pcd = point_cloud
        initial_points = len(cleaned_pcd.points)
        print(f"Initial points: {initial_points}")
        
        # Statistical outlier removal
        if remove_statistical_outliers and initial_points > 0:
            print(f"Applying statistical outlier removal...")
            cleaned_pcd, ind = cleaned_pcd.remove_statistical_outlier(
                nb_neighbors=statistical_nb_neighbors,
                std_ratio=statistical_std_ratio
            )
            print(f"  ✓ Removed {initial_points - len(cleaned_pcd.points)} outliers")
            print(f"  ✓ Remaining points: {len(cleaned_pcd.points)}")
        
        # Radius outlier removal
        if remove_radius_outliers and len(cleaned_pcd.points) > 0:
            print(f"Applying radius outlier removal...")
            before = len(cleaned_pcd.points)
            cleaned_pcd, ind = cleaned_pcd.remove_radius_outlier(
                nb_points=radius_filter_nb_points,
                radius=radius_filter_radius
            )
            print(f"  ✓ Removed {before - len(cleaned_pcd.points)} sparse points")
            print(f"  ✓ Remaining points: {len(cleaned_pcd.points)}")
        
        # Voxel downsampling
        if downsample and len(cleaned_pcd.points) > 0:
            print(f"Applying voxel downsampling (voxel_size={voxel_size})...")
            before = len(cleaned_pcd.points)
            cleaned_pcd = cleaned_pcd.voxel_down_sample(voxel_size=voxel_size)
            print(f"  ✓ Downsampled from {before} to {len(cleaned_pcd.points)} points")
        
        # Estimate normals if not present
        if not cleaned_pcd.has_normals() and len(cleaned_pcd.points) > 0:
            print("Estimating normals...")
            cleaned_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=voxel_size * 2, max_nn=30
                )
            )
        
        print(f"✅ Final point cloud: {len(cleaned_pcd.points)} points")
        
        return cleaned_pcd
    
    # ==================== Utility Functions ====================
    
    def load_vggt_point_cloud(self, point_cloud_path: str) -> o3d.geometry.PointCloud:
        """Load VGGT point cloud output (supports .npy, .pkl, .npz, .ply, .pcd)."""
        print(f"Loading VGGT point cloud from {point_cloud_path}")
        
        path = Path(point_cloud_path)
        
        # If path is a directory, look for point cloud files
        if path.is_dir():
            print(f"⚠️  Path is a directory. Searching for point cloud files...")
            
            # Search for VGGT output files (prioritize .pkl and .npy from VGGT)
            for pattern in ['*.pkl', '*.npy', '*.npz', '*.ply', '*.pcd']:
                files = list(path.glob(pattern))
                if files:
                    # Use the first found file
                    path = files[0]
                    print(f"✓ Found point cloud file: {path.name}")
                    break
            else:
                raise ValueError(
                    f"No point cloud files found in directory: {point_cloud_path}\n"
                    f"Expected files with extensions: .pkl, .npy, .npz, .ply, .pcd"
                )
        
        if not path.exists():
            raise FileNotFoundError(f"Point cloud file not found: {path}")
        
        print(f"Loading file: {path.name} (format: {path.suffix})")
        
        # Load based on file format
        if path.suffix in ['.ply', '.pcd']:
            pcd = o3d.io.read_point_cloud(str(path))
            
        elif path.suffix == '.pkl':
            # VGGT pickle format
            data = None
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
            except (pickle.UnpicklingError, ModuleNotFoundError, AttributeError) as e:
                error_str = str(e)
                
                if "persistent_id" in error_str:
                    # This file contains torch.Tensor or numpy objects saved with custom serialization
                    print(f"  ⚠️  Standard pickle failed, this appears to be torch-serialized")
                    print(f"  Attempting torch.load()...")
                    
                    try:
                        data = torch.load(path, weights_only=False)
                        print(f"  ✓ Successfully loaded with torch.load()")
                    except Exception as e2:
                        print(f"  ⚠️  torch.load() failed: {str(e2)[:100]}")
                        
                        # Last resort: custom persistent_load handler
                        print(f"  Attempting custom unpickler with persistent_id handler...")
                        try:
                            def persistent_load(pid):
                                # Return None for persistent IDs as fallback
                                return None
                            
                            with open(path, 'rb') as f:
                                unpickler = pickle.Unpickler(f)
                                unpickler.persistent_load = persistent_load
                                data = unpickler.load()
                            
                            print(f"  ⚠️  Loaded with fallback handler (may lose torch tensor data)")
                        except Exception as e3:
                            raise RuntimeError(
                                f"\n❌ Failed to load pickle file '{path.name}':\n"
                                f"This appears to be a torch-serialized pickle file.\n\n"
                                f"Attempted 3 loading strategies (all failed):\n"
                                f"  1. Standard pickle.load()\n"
                                f"  2. torch.load() with weights_only=False\n"
                                f"  3. Custom persistent_load handler\n\n"
                                f"Troubleshooting:\n"
                                f"  • Ensure torch is installed: pip install torch\n"
                                f"  • Try converting the file manually:\n"
                                f"    import torch\n"
                                f"    data = torch.load('{path}')\n"
                                f"    import pickle\n"
                                f"    pickle.dump(data, open('{path.stem}_converted.pkl', 'wb'))\n"
                            )
                else:
                    # Other pickle errors
                    raise RuntimeError(
                        f"Failed to load pickle file '{path.name}':\n{e}\n"
                        f"This may be caused by incompatible format, missing dependencies, or corruption."
                    )
            
            if data is None:
                raise RuntimeError(f"Could not load pickle file: {path}")
            
            pcd = self._convert_vggt_data_to_pointcloud(data, 'pkl')
            
        elif path.suffix == '.npy':
            # VGGT numpy format
            data = np.load(path, allow_pickle=True)
            pcd = self._convert_vggt_data_to_pointcloud(data, 'npy')
            
        elif path.suffix == '.npz':
            # Compressed numpy format
            data = np.load(path)
            pcd = self._convert_vggt_data_to_pointcloud(data, 'npz')
            
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        print(f"✅ Loaded {len(pcd.points)} points")
        
        # Auto-save as PLY if source was not PLY
        if path.suffix not in ['.ply', '.pcd']:
            ply_path = self.pointclouds_dir / f"{path.stem}_converted.ply"
            o3d.io.write_point_cloud(str(ply_path), pcd)
            print(f"💾 Saved PLY version to: {ply_path.name}")
        
        return pcd
    
    def _convert_vggt_data_to_pointcloud(self, data, format_type: str) -> o3d.geometry.PointCloud:
        """Convert VGGT data structures to Open3D point cloud."""
        pcd = o3d.geometry.PointCloud()
        
        if format_type == 'pkl':
            # Handle pickle format - could be dict or direct array
            if isinstance(data, dict):
                # Try common keys first
                if 'world_points' in data:
                    points = np.array(data['world_points'])
                    print(f"  Using 'world_points' as point coordinates")
                    # Try to use confidence as colors
                    if 'world_points_conf' in data:
                        conf = np.array(data['world_points_conf'], dtype=np.float32)
                        if conf.ndim == 1:
                            # Normalize confidence to 0-1 first
                            if conf.max() > 1.0:
                                conf = conf / 255.0
                            # Repeat to RGB
                            conf = np.repeat(conf[:, np.newaxis], 3, axis=1).astype(np.float32)
                        else:
                            # Already 2D, just normalize
                            if conf.max() > 1.0:
                                conf = conf / 255.0
                            conf = conf.astype(np.float32)
                        
                        # Clamp to [0, 1]
                        conf = np.clip(conf, 0.0, 1.0)
                        pcd.colors = o3d.utility.Vector3dVector(conf)
                elif 'points' in data:
                    points = np.array(data['points'])
                elif 'xyz' in data:
                    points = np.array(data['xyz'])
                elif 'vertices' in data:
                    points = np.array(data['vertices'])
                elif 'point_cloud' in data:
                    points = np.array(data['point_cloud'])
                else:
                    # Try to find array-like values with shape (N, 3)
                    for key, value in data.items():
                        if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == 3:
                            points = value
                            print(f"  Using key '{key}' as point coordinates")
                            break
                    else:
                        raise ValueError(
                            f"Could not find 3D points in pickle data.\n"
                            f"Available keys: {list(data.keys())}\n"
                            f"Expected keys: 'world_points', 'points', 'xyz', 'vertices', or similar\n"
                            f"or any array with shape (N, 3)"
                        )
                
                # Try to extract colors if available
                if not pcd.has_colors():
                    if 'colors' in data:
                        colors = np.array(data['colors'])
                        if colors.max() > 1.0:
                            colors = colors / 255.0
                        pcd.colors = o3d.utility.Vector3dVector(colors)
                    elif 'rgb' in data:
                        colors = np.array(data['rgb'])
                        if colors.max() > 1.0:
                            colors = colors / 255.0
                        pcd.colors = o3d.utility.Vector3dVector(colors)
            else:
                # Direct array
                points = np.array(data)
                if points.ndim == 1:
                    points = points.reshape(-1, 3)
            
        elif format_type == 'npy':
            # Handle .npy format
            if isinstance(data, np.ndarray):
                if data.dtype == object:
                    # Might be a pickled object in npy
                    data = data.item()
                    return self._convert_vggt_data_to_pointcloud(data, 'pkl')
                else:
                    points = data
                    if points.ndim == 1 and points.shape[0] % 3 == 0:
                        points = points.reshape(-1, 3)
            else:
                raise ValueError(f"Unexpected npy data type: {type(data)}")
        
        elif format_type == 'npz':
            # Handle .npz format
            if 'world_points' in data:
                points = data['world_points']
                print(f"  Using 'world_points' from npz")
            elif 'points' in data:
                points = data['points']
            elif 'xyz' in data:
                points = data['xyz']
            else:
                # Use first array that looks like 3D points
                for key in data.files:
                    arr = data[key]
                    if arr.ndim == 2 and arr.shape[1] == 3:
                        points = arr
                        print(f"  Using key '{key}' as point coordinates")
                        break
                else:
                    raise ValueError(f"Could not find 3D points in npz data. Files: {data.files}")
            
            if 'colors' in data:
                colors = data['colors']
                if colors.max() > 1.0:
                    colors = colors / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Ensure points are Nx3
        if points.shape[1] != 3:
            raise ValueError(f"Expected Nx3 points, got shape {points.shape}")
        
        pcd.points = o3d.utility.Vector3dVector(points)
        
        return pcd
    
    def save_point_cloud(self, point_cloud: o3d.geometry.PointCloud, filename: str, format: Optional[str] = None) -> Path:
        """
        Save point cloud to file.
        
        Args:
            point_cloud: Point cloud to save
            filename: Output filename (extension determines format if format is None)
            format: Force specific format ('ply', 'pcd', 'xyz', 'xyzrgb', 'pts')
        """
        output_path = self.pointclouds_dir / filename
        
        # Override extension if format is specified
        if format:
            output_path = output_path.with_suffix(f'.{format}')
        
        # Ensure the format is supported
        if output_path.suffix.lower() not in ['.ply', '.pcd', '.xyz', '.xyzn', '.xyzrgb', '.pts']:
            print(f"⚠️  Format {output_path.suffix} not supported by Open3D, defaulting to .ply")
            output_path = output_path.with_suffix('.ply')
        
        o3d.io.write_point_cloud(str(output_path), point_cloud)
        print(f"✅ Saved point cloud to {output_path}")
        return output_path
    
    def convert_point_cloud_format(self, input_path: str, output_format: str = 'ply') -> Path:
        """
        Convert point cloud from one format to another.
        
        Args:
            input_path: Path to input point cloud
            output_format: Desired output format (ply, pcd, xyz, etc.)
            
        Returns:
            Path to converted file
        """
        print(f"Converting {input_path} to {output_format} format...")
        pcd = self.load_vggt_point_cloud(input_path)
        
        input_name = Path(input_path).stem
        output_filename = f"{input_name}.{output_format}"
        
        return self.save_point_cloud(pcd, output_filename, format=output_format)
    
    def visualize_point_cloud(self, point_cloud: o3d.geometry.PointCloud, window_name: str = "Point Cloud"):
        """Visualize point cloud with Open3D."""
        print(f"Visualizing {window_name}...")
        o3d.visualization.draw_geometries(
            [point_cloud],
            window_name=window_name,
            width=1280,
            height=720,
            point_show_normal=False
        )
    
    def compute_distances(
        self, 
        point_cloud: o3d.geometry.PointCloud,
        pairs: List[Tuple[int, int]]
    ) -> List[float]:
        """
        Compute distances between point pairs.
        
        Args:
            point_cloud: Input point cloud
            pairs: List of (index1, index2) tuples
            
        Returns:
            List of distances in meters (if scaled)
        """
        points = np.asarray(point_cloud.points)
        distances = []
        
        for i, j in pairs:
            dist = np.linalg.norm(points[i] - points[j])
            distances.append(dist)
            print(f"Distance between point {i} and {j}: {dist:.4f} m")
        
        return distances
    
    def print_summary(self):
        """Print pipeline summary."""
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print(f"\nGenerated subdirectories:")
        print(f"  📁 {self.colmap_dir.name}/")
        print(f"  📁 {self.depths_dir.name}/")
        print(f"  📁 {self.pointclouds_dir.name}/")
        print("\nOptional dependencies:")
        for dep, installed in OPTIONAL_DEPS.items():
            status = "✅" if installed else "⚠️"
            print(f"  {status} {dep}")


def main():
    parser = argparse.ArgumentParser(
        description="VGGT Point Cloud Refinement Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert VGGT output (.npy/.pkl) to PLY format
  python vggt_refinement_pipeline.py --scene_dir /path/to/scene --point_cloud output.npy --convert_format ply
  
  # Basic refinement with Bundle Adjustment
  python vggt_refinement_pipeline.py --scene_dir /path/to/scene --point_cloud output.ply --use_ba
  
  # With metric scaling using known distance
  python vggt_refinement_pipeline.py --scene_dir /path/to/scene --point_cloud output.pkl \\
    --known_distance 2.5 --point_indices 100 500
  
  # With GCP file
  python vggt_refinement_pipeline.py --scene_dir /path/to/scene --point_cloud output.npy \\
    --gcp_file ground_control_points.json --visualize
        """
    )
    
    # Input/Output
    parser.add_argument(
        "--scene_dir",
        type=str,
        required=True,
        help="Path to VGGT scene directory"
    )
    parser.add_argument(
        "--point_cloud",
        type=str,
        help="Path to VGGT point cloud file (if not using BA)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for refined results"
    )
    
    # Bundle Adjustment
    parser.add_argument(
        "--use_ba",
        action="store_true",
        default=False,
        help="Run bundle adjustment (requires demo_colmap.py from VGGT)"
    )
    parser.add_argument(
        "--demo_script",
        type=str,
        help="Path to demo_colmap.py script"
    )
    parser.add_argument(
        "--max_query_pts",
        type=int,
        default=4096,
        help="Max query points for BA (default: 4096, reduce for low-memory GPUs)"
    )
    parser.add_argument(
        "--query_frame_num",
        type=int,
        default=3,
        help="Number of query frames for BA (default: 3)"
    )
    parser.add_argument(
        "--reduce_memory",
        action="store_true",
        help="Enable memory reduction mode (ultra-low settings for very limited GPU memory)"
    )
    parser.add_argument(
        "--disable_fine_tracking",
        action="store_true",
        help="Disable fine tracking in BA (saves a lot of memory but reduces accuracy)"
    )
    parser.add_argument(
        "--memory_fraction",
        type=float,
        help="Limit GPU memory usage (0.0-1.0, e.g., 0.5 for 50%)"
    )
    
    # Metric Scaling
    parser.add_argument(
        "--gcp_file",
        type=str,
        help="Path to Ground Control Points JSON file"
    )
    parser.add_argument(
        "--known_distance",
        type=float,
        help="Known distance in meters (requires --point_indices)"
    )
    parser.add_argument(
        "--point_indices",
        type=int,
        nargs=2,
        help="Two point indices for known distance scaling"
    )
    
    # Post-Processing
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.01,
        help="Voxel size for downsampling (meters, default: 0.01)"
    )
    parser.add_argument(
        "--no_denoise",
        action="store_true",
        help="Skip denoising steps"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize results"
    )
    parser.add_argument(
        "--convert_format",
        type=str,
        choices=['ply', 'pcd', 'xyz', 'xyzrgb', 'pts'],
        help="Convert point cloud to specified format and save"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = VGGTRefinementPipeline(args.scene_dir, args.output_dir)
    
    # Step 1: Bundle Adjustment (optional)
    if args.use_ba:
        result = pipeline.run_bundle_adjustment(
            max_query_pts=args.max_query_pts,
            query_frame_num=args.query_frame_num,
            use_ba=True,
            demo_script_path=args.demo_script,
            reduce_memory=args.reduce_memory,
            disable_fine_tracking=args.disable_fine_tracking,
            memory_fraction=args.memory_fraction
        )
        if result is None:
            print("\n⚠️  Continuing without Bundle Adjustment...")
    
    # Load point cloud
    if args.point_cloud:
        point_cloud = pipeline.load_vggt_point_cloud(args.point_cloud)
        
        # If format conversion requested, do it and exit
        if args.convert_format:
            print(f"\n{'='*60}")
            print(f"FORMAT CONVERSION: {Path(args.point_cloud).suffix} → .{args.convert_format}")
            print(f"{'='*60}")
            output_path = pipeline.save_point_cloud(
                point_cloud, 
                f"converted.{args.convert_format}",
                format=args.convert_format
            )
            print(f"\n✅ Conversion complete!")
            print(f"📁 Output: {output_path}")
            return
    else:
        print("⚠️  No point cloud specified. Skipping to next steps.")
        pipeline.print_summary()
        return
    
    # Step 2: Metric Scaling
    if args.gcp_file:
        point_cloud, transform = pipeline.apply_metric_scaling(
            point_cloud,
            gcp_file=args.gcp_file
        )
        pipeline.save_point_cloud(point_cloud, "scaled_point_cloud.ply")
    elif args.known_distance and args.point_indices:
        points = np.asarray(point_cloud.points)
        p1 = points[args.point_indices[0]]
        p2 = points[args.point_indices[1]]
        point_cloud, transform = pipeline.apply_metric_scaling(
            point_cloud,
            known_distance=(p1, p2, args.known_distance)
        )
        pipeline.save_point_cloud(point_cloud, "scaled_point_cloud.ply")
    
    # Step 4: Post-Processing
    if not args.no_denoise:
        point_cloud = pipeline.post_process_point_cloud(
            point_cloud,
            voxel_size=args.voxel_size,
            remove_statistical_outliers=True,
            remove_radius_outliers=True,
            downsample=True
        )
        pipeline.save_point_cloud(point_cloud, "refined_point_cloud.ply")
    
    # Visualization
    if args.visualize:
        pipeline.visualize_point_cloud(point_cloud, "Refined Point Cloud")
    
    # Summary
    pipeline.print_summary()
    
    print("\n" + "=" * 60)
    print("REFINEMENT PIPELINE COMPLETED")
    print("=" * 60)
    print(f"✅ Final point cloud: {len(point_cloud.points)} points")
    print(f"📁 Output directory: {pipeline.output_dir}")


if __name__ == "__main__":
    main()
