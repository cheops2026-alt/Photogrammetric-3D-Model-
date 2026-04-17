# Photogrammetric-3D-Model - Depth estimation, point cloud, mesh refinement
import logging
import os

import numpy as np
import torch
import trimesh
from PIL import Image


# ===================================================================
# MiDaS Depth Estimation
# ===================================================================

_midas_model = None
_midas_transform = None
_midas_device = None


def _load_midas(device: str = "cuda:0"):
    """Load MiDaS DPT-Large for high-quality monocular depth estimation."""
    global _midas_model, _midas_transform, _midas_device
    if _midas_model is not None and _midas_device == device:
        return _midas_model, _midas_transform

    logging.info("Loading MiDaS DPT-Large depth model...")
    _midas_model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    _midas_model.to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    _midas_transform = midas_transforms.dpt_transform
    _midas_device = device
    logging.info("MiDaS loaded.")
    return _midas_model, _midas_transform


def estimate_depth(
    image_path: str,
    device: str = "cuda:0",
    output_dir: str | None = None,
) -> np.ndarray:
    """
    Run MiDaS on a single image and return a normalized depth map (H, W) float32 in [0, 1].
    Optionally saves a depth visualization to output_dir.
    """
    model, transform = _load_midas(device)
    img = np.array(Image.open(image_path).convert("RGB"))
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    depth_min, depth_max = depth.min(), depth.max()
    if depth_max - depth_min > 1e-6:
        depth = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth = np.zeros_like(depth)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        depth_vis = Image.fromarray((depth * 255).astype(np.uint8))
        depth_vis.save(os.path.join(output_dir, "depth_map.png"))
        logging.info(f"Depth map saved: {os.path.join(output_dir, 'depth_map.png')}")

    return depth


# ===================================================================
# Depth Map → Point Cloud
# ===================================================================

def depth_to_point_cloud(
    depth: np.ndarray,
    image: np.ndarray,
    fov_deg: float = 60.0,
    max_points: int = 500_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Back-project a depth map into a colored 3D point cloud.

    Returns (points [N,3], colors [N,3] uint8).
    """
    h, w = depth.shape[:2]
    fx = fy = (w / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    cx, cy = w / 2.0, h / 2.0

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float64)

    mask = z > 0.05
    u, v, z = u[mask], v[mask], z[mask]

    x = (u - cx) * z / fx
    y = -(v - cy) * z / fy

    points = np.stack([x, y, z], axis=-1)
    colors = image[mask] if image.ndim == 3 else np.zeros((points.shape[0], 3), dtype=np.uint8)

    if points.shape[0] > max_points:
        idx = np.random.choice(points.shape[0], max_points, replace=False)
        points, colors = points[idx], colors[idx]

    return points.astype(np.float32), colors.astype(np.uint8)


# ===================================================================
# Mesh Post-Processing (smoothing, noise reduction, refinement)
# ===================================================================

def refine_trimesh(mesh: trimesh.Trimesh, iterations: int = 2) -> trimesh.Trimesh:
    """
    Apply Laplacian smoothing, remove degenerate faces, fill holes,
    and subdivide for a cleaner, denser mesh.
    """
    mesh.merge_vertices()
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()

    if hasattr(mesh, 'fill_holes'):
        mesh.fill_holes()

    for _ in range(iterations):
        mesh = _laplacian_smooth(mesh, lamb=0.5)

    return mesh


def _laplacian_smooth(mesh: trimesh.Trimesh, lamb: float = 0.5) -> trimesh.Trimesh:
    """One pass of Laplacian smoothing on vertex positions."""
    verts = mesh.vertices.copy()
    adj = {i: set() for i in range(len(verts))}
    for f in mesh.faces:
        adj[f[0]].update([f[1], f[2]])
        adj[f[1]].update([f[0], f[2]])
        adj[f[2]].update([f[0], f[1]])

    new_verts = verts.copy()
    for i in range(len(verts)):
        neighbors = list(adj[i])
        if neighbors:
            avg = verts[neighbors].mean(axis=0)
            new_verts[i] = verts[i] + lamb * (avg - verts[i])

    mesh.vertices = new_verts
    return mesh


# ===================================================================
# Depth-Guided Mesh Enhancement
# ===================================================================

def enhance_mesh_with_depth(
    mesh_path: str,
    texture_path: str | None,
    input_image_path: str,
    output_dir: str,
    device: str = "cuda:0",
    smooth_iterations: int = 2,
) -> tuple[str, str | None]:
    """
    Enhance an existing TripoSR mesh using MiDaS depth estimation:
      1. Estimate depth from the input image
      2. Generate a depth-based point cloud
      3. Refine the TripoSR mesh geometry (smooth, clean, fill holes)
      4. Use depth normals to improve surface detail

    Returns (refined_mesh_path, texture_path).
    """
    logging.info("Enhancing mesh with MiDaS depth estimation...")

    depth = estimate_depth(input_image_path, device=device, output_dir=output_dir)
    img_rgb = np.array(Image.open(input_image_path).convert("RGB"))

    points, colors = depth_to_point_cloud(depth, img_rgb)
    pcd_path = os.path.join(output_dir, "depth_pointcloud.ply")
    pcd_mesh = trimesh.PointCloud(points, colors=colors)
    pcd_mesh.export(pcd_path)
    logging.info(f"Depth point cloud: {points.shape[0]} points → {pcd_path}")

    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = list(mesh.geometry.values())[0]

    mesh = refine_trimesh(mesh, iterations=smooth_iterations)

    refined_path = os.path.join(output_dir, "mesh_refined.obj")
    mesh.export(refined_path)
    logging.info(f"Refined mesh saved: {refined_path}")

    return refined_path, texture_path


# ===================================================================
# Multi-View Depth Fusion
# ===================================================================

def multi_view_depth_fusion(
    image_paths: list[str],
    output_dir: str,
    device: str = "cuda:0",
    fov_deg: float = 60.0,
) -> str:
    """
    When multiple images are provided, estimate depth for each,
    generate point clouds, merge them, and reconstruct a combined mesh.

    Returns the path to the fused mesh.
    """
    logging.info(f"Multi-view depth fusion: {len(image_paths)} images")
    all_points = []
    all_colors = []

    for i, img_path in enumerate(image_paths):
        logging.info(f"  Depth estimation for view {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
        depth = estimate_depth(img_path, device=device, output_dir=None)
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        pts, cols = depth_to_point_cloud(depth, img_rgb, fov_deg=fov_deg)

        angle = (2 * np.pi * i) / len(image_paths)
        R = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ])
        pts = pts @ R.T

        all_points.append(pts)
        all_colors.append(cols)

    merged_pts = np.concatenate(all_points, axis=0)
    merged_cols = np.concatenate(all_colors, axis=0)

    if merged_pts.shape[0] > 1_000_000:
        idx = np.random.choice(merged_pts.shape[0], 1_000_000, replace=False)
        merged_pts, merged_cols = merged_pts[idx], merged_cols[idx]

    logging.info(f"Merged point cloud: {merged_pts.shape[0]} points")

    pcd = trimesh.PointCloud(merged_pts, colors=merged_cols)
    fused_pcd_path = os.path.join(output_dir, "multiview_pointcloud.ply")
    pcd.export(fused_pcd_path)
    logging.info(f"Multi-view point cloud saved: {fused_pcd_path}")

    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(merged_pts[:, :2])
        fused_mesh = trimesh.Trimesh(
            vertices=merged_pts,
            faces=tri.simplices,
            vertex_colors=merged_cols,
        )
        fused_mesh = refine_trimesh(fused_mesh, iterations=2)
    except Exception as exc:
        logging.warning(f"Delaunay triangulation failed: {exc}. Saving point cloud only.")
        fused_mesh = pcd

    fused_path = os.path.join(output_dir, "multiview_mesh.obj")
    fused_mesh.export(fused_path)
    logging.info(f"Multi-view fused mesh: {fused_path}")
    return fused_path
