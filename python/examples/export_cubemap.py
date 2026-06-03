#!/usr/bin/env python3
"""
Export a COLMAP reconstruction with equirectangular (ERP) cameras to a
pinhole cubemap dataset that gsplat can consume.

For each ERP image, this script generates 6 cubemap faces with 90-degree FOV
pinhole cameras. The face poses are derived by composing the original ERP
camera pose with the standard cubemap face rotations.

Output layout:
    output_dir/
      images/
        <base_name>_front.png
        <base_name>_right.png
        <base_name>_back.png
        <base_name>_left.png
        <base_name>_top.png
        <base_name>_bottom.png
      sparse/
        cameras.txt
        images.txt
        points3D.txt

Usage:
    python export_cubemap.py \
        --input_model /path/to/erp_sparse_model \
        --input_images /path/to/erp_images \
        --output_dir /path/to/cubemap_output \
        --face_size 512
"""

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# COLMAP text model I/O
# ---------------------------------------------------------------------------


def read_cameras_txt(path: Path) -> dict[int, dict]:
    """Read COLMAP cameras.txt. Returns {camera_id: camera_dict}."""
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]
            cameras[cam_id] = {
                "id": cam_id,
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def read_images_txt(path: Path) -> dict[int, dict]:
    """Read COLMAP images.txt. Returns {image_id: image_dict}.

    Each image occupies two lines (a header line and a POINTS2D line). The
    POINTS2D line may be empty for images without observations, so we must not
    drop blank lines while pairing.
    """
    images = {}
    with open(path) as f:
        raw = [l.rstrip("\n") for l in f if not l.lstrip().startswith("#")]
    expect_header = True
    for line in raw:
        if expect_header:
            if not line.strip():
                continue
            header = line.split()
            img_id = int(header[0])
            qvec = np.array([float(x) for x in header[1:5]])
            tvec = np.array([float(x) for x in header[5:8]])
            cam_id = int(header[8])
            name = header[9]
            images[img_id] = {
                "id": img_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": cam_id,
                "name": name,
            }
            expect_header = False
        else:
            # POINTS2D line (possibly empty); not needed for the export.
            expect_header = True
    return images


def read_points3D_txt(path: Path) -> dict[int, dict]:
    """Read COLMAP points3D.txt. Returns {point3D_id: point_dict}."""
    points = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pt_id = int(parts[0])
            xyz = np.array([float(x) for x in parts[1:4]])
            rgb = np.array([int(x) for x in parts[4:7]])
            error = float(parts[7])
            track = []
            track_parts = parts[8:]
            for i in range(0, len(track_parts) - 1, 2):
                track.append((int(track_parts[i]), int(track_parts[i + 1])))
            points[pt_id] = {
                "id": pt_id,
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "track": track,
            }
    return points


def write_cameras_txt(path: Path, cameras: dict[int, dict]) -> None:
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cam in cameras.values():
            params_str = " ".join(f"{p:.12g}" for p in cam["params"])
            f.write(
                f"{cam['id']} {cam['model']} {cam['width']} {cam['height']} "
                f"{params_str}\n"
            )


def write_images_txt(path: Path, images: dict[int, dict]) -> None:
    with open(path, "w") as f:
        f.write("# Image list with one line of data per image:\n")
        f.write(
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, "
            "CAMERA_ID, NAME\n"
        )
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for img in images.values():
            q = img["qvec"]
            t = img["tvec"]
            f.write(
                f"{img['id']} {q[0]:.12g} {q[1]:.12g} {q[2]:.12g} {q[3]:.12g} "
                f"{t[0]:.12g} {t[1]:.12g} {t[2]:.12g} "
                f"{img['camera_id']} {img['name']}\n"
            )
            points2D = img.get("points2D", [])
            f.write(
                " ".join(
                    f"{x:.6f} {y:.6f} {pt_id}" for x, y, pt_id in points2D
                )
                + "\n"
            )


def write_points3D_txt(path: Path, points: dict[int, dict]) -> None:
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write(
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
            "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        )
        for pt in points.values():
            xyz = pt["xyz"]
            rgb = pt["rgb"]
            track = pt.get("track_out", [])
            track_str = " ".join(
                f"{img_id} {point2D_idx}" for img_id, point2D_idx in track
            )
            f.write(
                f"{pt['id']} {xyz[0]:.12g} {xyz[1]:.12g} {xyz[2]:.12g} "
                f"{rgb[0]} {rgb[1]} {rgb[2]} {pt['error']:.12g}"
                + (f" {track_str}" if track_str else "")
                + "\n"
            )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP quaternion (qw, qx, qy, qz) to rotation matrix."""
    qw, qx, qy, qz = qvec
    R = np.array(
        [
            [
                1 - 2 * (qy**2 + qz**2),
                2 * (qx * qy - qw * qz),
                2 * (qx * qz + qw * qy),
            ],
            [
                2 * (qx * qy + qw * qz),
                1 - 2 * (qx**2 + qz**2),
                2 * (qy * qz - qw * qx),
            ],
            [
                2 * (qx * qz - qw * qy),
                2 * (qy * qz + qw * qx),
                1 - 2 * (qx**2 + qy**2),
            ],
        ],
        dtype=np.float64,
    )
    return R


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to COLMAP quaternion (qw, qx, qy, qz).

    Uses the eigendecomposition method from COLMAP's read_write_model.py, which
    is the exact inverse of ``qvec2rotmat``.
    """
    R = np.array(R, dtype=np.float64)
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


# Face rotations that map the standard pinhole frame (+Z forward, +X right,
# +Y down) to each cubemap face direction.
FACE_NAMES = ["front", "right", "back", "left", "top", "bottom"]

# Each column is where the camera axis maps in world space:
#   col 0 = R * [1,0,0] = where camera +X (right) goes
#   col 1 = R * [0,1,0] = where camera +Y (down) goes
#   col 2 = R * [0,0,1] = where camera +Z (forward) goes
FACE_ROTATIONS = {
    "front": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
    "right": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64),
    "back": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64),
    "left": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
    "top": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64),
    "bottom": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64),
}


def compose_pose(
    face_rot: np.ndarray, qvec_erp: np.ndarray, tvec_erp: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compose ERP pose with face rotation.

    ``face_rot`` is ``erp_from_face`` (it maps a face-frame ray to the ERP
    frame, see ``render_cubemap_face``). The face camera shares the ERP camera
    center, so the face extrinsics are::

        face_from_world = face_from_erp @ erp_from_world = face_rot.T @ R_erp
        t_face          = face_rot.T @ t_erp
    """
    R_erp = qvec2rotmat(qvec_erp)
    R_face = face_rot.T @ R_erp
    t_face = face_rot.T @ tvec_erp
    qvec_face = rotmat2qvec(R_face)
    # Normalize quaternion to unit length
    qvec_face /= np.linalg.norm(qvec_face)
    # Ensure positive real part (COLMAP convention)
    if qvec_face[0] < 0:
        qvec_face = -qvec_face
    return qvec_face, t_face


# ---------------------------------------------------------------------------
# Cubemap rendering
# ---------------------------------------------------------------------------


def erp_img_from_cam(
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rays: np.ndarray,
) -> np.ndarray:
    """
    Project camera rays into an ERP image.

    rays: (N, 3) array of unit rays in the ERP camera frame.
    Returns: (N, 2) array of (u, v) pixel coordinates.
    """
    x = rays[:, 0]
    y = rays[:, 1]
    z = rays[:, 2]
    lon = np.arctan2(x, z)
    lat = np.arctan2(-y, np.linalg.norm(rays[:, [0, 2]], axis=1))
    u = cx + fx * lon
    v = cy - fy * lat
    return np.stack([u, v], axis=-1)


def render_cubemap_face(
    pano_image: np.ndarray,
    face_rot: np.ndarray,
    face_size: int,
    fx_erp: float,
    fy_erp: float,
    cx_erp: float,
    cy_erp: float,
) -> np.ndarray:
    """
    Render a single cubemap face from an ERP panorama.

    face_rot: rotation that maps a face-frame ray to the ERP camera frame
              (i.e. ``erp_from_face``). Applied directly to each face ray.
    """
    pano_h, pano_w = pano_image.shape[:2]

    # Build pinhole rays: for each pixel, ray = normalize((u, v, 1))
    x = np.arange(face_size, dtype=np.float32) + 0.5
    y = np.arange(face_size, dtype=np.float32) + 0.5
    xv, yv = np.meshgrid(x, y)
    fx_face = face_size / 2.0
    fy_face = face_size / 2.0
    cx_face = face_size / 2.0
    cy_face = face_size / 2.0
    u = (xv - cx_face) / fx_face
    v = (yv - cy_face) / fy_face
    rays_face = np.stack([u, v, np.ones_like(u)], axis=-1)
    rays_face /= np.linalg.norm(rays_face, axis=-1, keepdims=True)

    # Transform rays to ERP camera frame
    rays_face = rays_face.reshape(-1, 3)
    rays_erp = rays_face @ face_rot.T  # face_rot^{-1} = face_rot^T

    # Project to ERP image
    uv = erp_img_from_cam(pano_w, pano_h, fx_erp, fy_erp, cx_erp, cy_erp, rays_erp)
    uv = uv.reshape(face_size, face_size, 2).astype(np.float32)

    # OpenCV remap uses pixel-center origin, so subtract 0.5
    # Wait, erp_img_from_cam already returns pixel-center coordinates.
    # cv2.remap expects source coordinates in pixel units.
    # The COLMAP convention uses (0.5, 0.5) as upper-left pixel center,
    # and cv2.remap also samples at pixel centers when given integer coords.
    # So no adjustment needed for the coordinate system.
    map_x = uv[..., 0]
    map_y = uv[..., 1]

    face_image = cv2.remap(
        pano_image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
    return face_image


# ---------------------------------------------------------------------------
# Main export logic
# ---------------------------------------------------------------------------


def export_cubemap(
    input_model_dir: Path,
    input_image_dir: Path,
    output_dir: Path,
    face_size: int,
) -> None:
    # ------------------------------------------------------------------
    # Read input model
    # ------------------------------------------------------------------
    cameras = read_cameras_txt(input_model_dir / "cameras.txt")
    images = read_images_txt(input_model_dir / "images.txt")
    points3D = read_points3D_txt(input_model_dir / "points3D.txt")

    # ------------------------------------------------------------------
    # Validate and identify ERP cameras
    # ------------------------------------------------------------------
    erp_camera_ids = set()
    for cam_id, cam in cameras.items():
        model = cam["model"]
        if model in ("EQUIRECTANGULAR", "SPHERE"):
            erp_camera_ids.add(cam_id)
            if len(cam["params"]) != 4:
                raise ValueError(
                    f"ERP camera {cam_id} must have 4 params "
                    f"(fx, fy, cx, cy), got {cam['params']}"
                )
        else:
            print(
                f"Warning: camera {cam_id} has model '{model}'; "
                f"only ERP cameras are converted to cubemap faces."
            )

    if not erp_camera_ids:
        raise ValueError("No equirectangular cameras found in the model.")

    # ------------------------------------------------------------------
    # Prepare output directories
    # ------------------------------------------------------------------
    output_images_dir = output_dir / "images"
    output_sparse_dir = output_dir / "sparse" / "0"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_sparse_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Create output cameras (one pinhole per face type)
    # ------------------------------------------------------------------
    # We create 6 camera IDs, one for each face orientation.
    # All faces share the same intrinsic parameters.
    max_cam_id = max(cameras.keys()) if cameras else 0
    output_cameras = {}
    face_camera_ids = {}
    fx_face = face_size / 2.0
    fy_face = face_size / 2.0
    cx_face = face_size / 2.0
    cy_face = face_size / 2.0
    for idx, face_name in enumerate(FACE_NAMES):
        cam_id = max_cam_id + 1 + idx
        face_camera_ids[face_name] = cam_id
        output_cameras[cam_id] = {
            "id": cam_id,
            "model": "PINHOLE",
            "width": face_size,
            "height": face_size,
            "params": [fx_face, fy_face, cx_face, cy_face],
        }

    # ------------------------------------------------------------------
    # Generate faces and output images
    # ------------------------------------------------------------------
    max_img_id = max(images.keys()) if images else 0
    output_images = {}
    img_id_counter = max_img_id + 1
    # Map each ERP image id to its generated (face_name, output_image_id) faces.
    erp_to_faces: dict[int, list[tuple[str, int]]] = {}

    for img_id, img in tqdm(images.items(), desc="Processing images"):
        cam_id = img["camera_id"]
        if cam_id not in erp_camera_ids:
            # Non-ERP image: copy unchanged
            img.setdefault("points2D", [])
            output_images[img_id] = img
            src_path = input_image_dir / img["name"]
            dst_path = output_images_dir / img["name"]
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
            continue

        # ERP image: generate 6 cubemap faces
        cam = cameras[cam_id]
        fx_erp, fy_erp, cx_erp, cy_erp = cam["params"]
        pano_path = input_image_dir / img["name"]
        if not pano_path.exists():
            print(f"Warning: image not found: {pano_path}, skipping.")
            continue

        pano_image = np.array(Image.open(pano_path).convert("RGB"))
        base_name = Path(img["name"]).stem

        for face_name in FACE_NAMES:
            face_rot = FACE_ROTATIONS[face_name]
            face_image = render_cubemap_face(
                pano_image,
                face_rot,
                face_size,
                fx_erp,
                fy_erp,
                cx_erp,
                cy_erp,
            )
            face_name_file = f"{base_name}_{face_name}.png"
            face_path = output_images_dir / face_name_file
            Image.fromarray(face_image).save(face_path)

            qvec_face, tvec_face = compose_pose(
                face_rot, img["qvec"], img["tvec"]
            )
            output_images[img_id_counter] = {
                "id": img_id_counter,
                "qvec": qvec_face,
                "tvec": tvec_face,
                "camera_id": face_camera_ids[face_name],
                "name": face_name_file,
                "points2D": [],
            }
            erp_to_faces.setdefault(img_id, []).append(
                (face_name, img_id_counter)
            )
            img_id_counter += 1

    # ------------------------------------------------------------------
    # Project 3D points into the cubemap faces to build tracks
    # ------------------------------------------------------------------
    # Each point is re-projected into the faces of every ERP image that
    # originally observed it. This preserves visibility while producing a fully
    # linked model (images <-> points2D <-> points3D tracks).
    for pt in tqdm(points3D.values(), desc="Building tracks"):
        xyz = pt["xyz"]
        pt["track_out"] = []
        for erp_img_id, _ in pt.get("track", []):
            for _, out_img_id in erp_to_faces.get(erp_img_id, []):
                face_img = output_images[out_img_id]
                R_face = qvec2rotmat(face_img["qvec"])
                p_cam = R_face @ xyz + face_img["tvec"]
                if p_cam[2] <= 0:
                    continue
                u = fx_face * p_cam[0] / p_cam[2] + cx_face
                v = fy_face * p_cam[1] / p_cam[2] + cy_face
                if not (0.0 <= u < face_size and 0.0 <= v < face_size):
                    continue
                point2D_idx = len(face_img["points2D"])
                face_img["points2D"].append((u, v, pt["id"]))
                pt["track_out"].append((out_img_id, point2D_idx))

    # ------------------------------------------------------------------
    # Write output model
    # ------------------------------------------------------------------
    write_cameras_txt(output_sparse_dir / "cameras.txt", output_cameras)
    write_images_txt(output_sparse_dir / "images.txt", output_images)
    write_points3D_txt(output_sparse_dir / "points3D.txt", points3D)

    print(f"\nExported {len(output_images)} images to {output_dir}")
    print(f"  Cameras: {len(output_cameras)} (6 cubemap face types)")
    print(f"  Points3D: {len(points3D)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export COLMAP ERP reconstruction to pinhole cubemap faces "
        "for gsplat training."
    )
    parser.add_argument(
        "--input_model",
        type=Path,
        required=True,
        help="Path to COLMAP sparse model directory (containing cameras.txt, "
        "images.txt, points3D.txt)",
    )
    parser.add_argument(
        "--input_images",
        type=Path,
        required=True,
        help="Path to directory containing the ERP images referenced in the model",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for the cubemap dataset",
    )
    parser.add_argument(
        "--face_size",
        type=int,
        default=512,
        help="Size of each cubemap face in pixels (default: 512)",
    )
    args = parser.parse_args()

    export_cubemap(
        args.input_model, args.input_images, args.output_dir, args.face_size
    )


if __name__ == "__main__":
    main()
