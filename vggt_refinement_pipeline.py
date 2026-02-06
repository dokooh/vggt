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
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import struct
import cv2


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
        
    # ==================== STEP 1: Bundle Adjustment ====================
    
    def run_bundle_adjustment(
        self, 
        max_query_pts: int = 4096,
        query_frame_num: int = 3,
        use_ba: bool = True
    ) -> Path:
        """
        Run Bundle Adjustment using VGGT's demo_colmap.py script.
        
        Args:
            max_query_pts: Maximum query points for BA
            query_frame_num: Number of query frames
            use_ba: Enable bundle adjustment
            
        Returns:
            Path to COLMAP output directory
        """
        print("=" * 60)
        print("STEP 1: Running Bundle Adjustment")
        print("=" * 60)
        
        colmap_dir = self.output_dir / "colmap_ba"
        colmap_dir.mkdir(exist_ok=True)
        
        # Construct command for VGGT's BA script
        cmd = [
            "python", "demo_colmap.py",
            f"--scene_dir={self.scene_dir}",
            f"--max_query_pts={max_query_pts}",
            f"--query_frame_num={query_frame_num}",
        ]
        
        if use_ba:
            cmd.append("--use_ba")
        
        print(f"Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Bundle Adjustment completed successfully!")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error running Bundle Adjustment: {e}")
            print(f"stderr: {e.stderr}")
            raise
        
        return colmap_dir
    
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
            
            print(f"Computed scale factor: {scale:.6f}")
            print(f"Current distance: {current_dist:.3f} -> Actual: {actual_dist:.3f}m")
            
        elif gcp_file is not None:
            # Use GCPs for similarity transformation
            gcps = self.load_ground_control_points(gcp_file)
            transform = self._compute_similarity_transform_from_gcps(point_cloud, gcps)
        else:
            print("Warning: No scaling information provided. Skipping metric scaling.")
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
        # This is a simplified version. For production, use COLMAP's alignment tools
        # or implement robust estimation with RANSAC
        
        if len(gcps) < 3:
            raise ValueError("Need at least 3 GCPs for similarity transform")
        
        # Extract point correspondences
        src_points = []
        dst_points = []
        
        for gcp in gcps:
            # Find closest point in cloud to pixel coordinate
            # (This is simplified - in practice, use camera projection)
            world_coord = np.array(gcp['world_coords'])
            dst_points.append(world_coord)
            
            # Placeholder: In real implementation, project pixel to 3D using camera params
            # For now, assume we have approximate 3D positions
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
        
        print(f"Similarity transform computed:")
        print(f"  Scale: {scale:.6f}")
        print(f"  Translation: {t}")
        
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
        # extrinsic: world to camera, so we need its inverse
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
            
            print(f"Unprojected depth map {i+1}/{len(depth_maps)}: {len(pcd.points)} points")
        
        # Combine all points
        fused_pcd = o3d.geometry.PointCloud()
        fused_pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
        
        if all_colors:
            fused_pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
        
        print(f"Fused point cloud: {len(fused_pcd.points)} total points")
        
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
        if remove_statistical_outliers:
            print(f"Applying statistical outlier removal...")
            cleaned_pcd, ind = cleaned_pcd.remove_statistical_outlier(
                nb_neighbors=statistical_nb_neighbors,
                std_ratio=statistical_std_ratio
            )
            print(f"  Removed {initial_points - len(cleaned_pcd.points)} outliers")
            print(f"  Remaining points: {len(cleaned_pcd.points)}")
        
        # Radius outlier removal
        if remove_radius_outliers:
            print(f"Applying radius outlier removal...")
            before = len(cleaned_pcd.points)
            cleaned_pcd, ind = cleaned_pcd.remove_radius_outlier(
                nb_points=radius_filter_nb_points,
                radius=radius_filter_radius
            )
            print(f"  Removed {before - len(cleaned_pcd.points)} sparse points")
            print(f"  Remaining points: {len(cleaned_pcd.points)}")
        
        # Voxel downsampling
        if downsample:
            print(f"Applying voxel downsampling (voxel_size={voxel_size})...")
            before = len(cleaned_pcd.points)
            cleaned_pcd = cleaned_pcd.voxel_down_sample(voxel_size=voxel_size)
            print(f"  Downsampled from {before} to {len(cleaned_pcd.points)} points")
        
        # Estimate normals if not present
        if not cleaned_pcd.has_normals():
            print("Estimating normals...")
            cleaned_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=voxel_size * 2, max_nn=30
                )
            )
        
        print(f"Final point cloud: {len(cleaned_pcd.points)} points")
        
        return cleaned_pcd
    
    # ==================== Utility Functions ====================
    
    def load_vggt_point_cloud(self, point_cloud_path: str) -> o3d.geometry.PointCloud:
        """Load VGGT point cloud output."""
        print(f"Loading VGGT point cloud from {point_cloud_path}")
        
        # VGGT may output .ply, .pcd, or numpy arrays
        path = Path(point_cloud_path)
        
        if path.suffix in ['.ply', '.pcd']:
            pcd = o3d.io.read_point_cloud(str(path))
        elif path.suffix in ['.npy', '.npz']:
            data = np.load(path)
            pcd = o3d.geometry.PointCloud()
            if isinstance(data, np.ndarray):
                pcd.points = o3d.utility.Vector3dVector(data)
            else:  # npz
                pcd.points = o3d.utility.Vector3dVector(data['points'])
                if 'colors' in data:
                    pcd.colors = o3d.utility.Vector3dVector(data['colors'])
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        print(f"Loaded {len(pcd.points)} points")
        return pcd
    
    def save_point_cloud(self, point_cloud: o3d.geometry.PointCloud, filename: str):
        """Save point cloud to file."""
        output_path = self.output_dir / filename
        o3d.io.write_point_cloud(str(output_path), point_cloud)
        print(f"Saved point cloud to {output_path}")
    
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


def main():
    parser = argparse.ArgumentParser(
        description="VGGT Point Cloud Refinement Pipeline"
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
        default=True,
        help="Run bundle adjustment"
    )
    parser.add_argument(
        "--max_query_pts",
        type=int,
        default=4096,
        help="Max query points for BA"
    )
    parser.add_argument(
        "--query_frame_num",
        type=int,
        default=3,
        help="Number of query frames for BA"
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
        help="Voxel size for downsampling (meters)"
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
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = VGGTRefinementPipeline(args.scene_dir, args.output_dir)
    
    # Step 1: Bundle Adjustment (optional)
    if args.use_ba:
        colmap_dir = pipeline.run_bundle_adjustment(
            max_query_pts=args.max_query_pts,
            query_frame_num=args.query_frame_num,
            use_ba=True
        )
        # After BA, load the refined point cloud from COLMAP output
        # (This would require reading COLMAP's points3D.bin - see helper below)
        print(f"Bundle adjustment completed. COLMAP output in {colmap_dir}")
        print("Load refined point cloud from COLMAP output (points3D.bin)")
    
    # Load point cloud
    if args.point_cloud:
        point_cloud = pipeline.load_vggt_point_cloud(args.point_cloud)
    else:
        print("Warning: No point cloud specified. Skipping to next steps.")
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
    
    print("\n" + "=" * 60)
    print("REFINEMENT PIPELINE COMPLETED")
    print("=" * 60)
    print(f"Final point cloud: {len(point_cloud.points)} points")
    print(f"Output directory: {pipeline.output_dir}")


if __name__ == "__main__":
    main()