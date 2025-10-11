import random
import time
import argparse
from pathlib import Path
from typing import List

import imageio.v3 as iio
import numpy as np
import tyro
from tqdm.auto import tqdm

import viser
import viser.transforms as vtf
from viser.extras.colmap import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)

parser = argparse.ArgumentParser(description="VGGT-X demo with viser for 3D visualization")
parser.add_argument(
    "--colmap_path", "-c", type=str, default="path/to/scene/sparse/0", help="Path to predicted colmap results")
parser.add_argument(
    "--colmap_ref_path", "-r", type=str, default=None, help="Path to ground truth colmap results")
parser.add_argument(
    "--images_path", "-i", type=str, default=None, help="Path to the images")
parser.add_argument("--downsample_factor", type=int, default=2, help="Downsample factor for the images")
parser.add_argument("--no_reorient", action="store_true", help="Do not reorient the scene based on average camera up direction")

def main() -> None:
    args = parser.parse_args()
    args.colmap_path = Path(args.colmap_path)
    args.colmap_ref_path = Path(args.colmap_ref_path) if args.colmap_ref_path else None
    args.images_path = Path(args.images_path) if args.images_path else None

    server = viser.ViserServer()
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Load the colmap info.
    cameras = read_cameras_binary(args.colmap_path / "cameras.bin")
    images = read_images_binary(args.colmap_path / "images.bin")
    points3d = read_points3d_binary(args.colmap_path / "points3D.bin")

    cameras_ref = read_cameras_binary(args.colmap_ref_path / "cameras.bin") if args.colmap_ref_path else None
    images_ref = read_images_binary(args.colmap_ref_path / "images.bin") if args.colmap_ref_path else None

    points = np.array([points3d[p_id].xyz for p_id in points3d])
    colors = np.array([points3d[p_id].rgb for p_id in points3d])

    gui_reset_up = server.gui.add_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )

    # Let's rotate the scene so the average camera direction is pointing up.
    if not args.no_reorient:
        average_up = (
            vtf.SO3(np.array([img.qvec for img in images.values()]))
            @ np.array([0.0, -1.0, 0.0])  # -y is up in the local frame!
        ).mean(axis=0)
        average_up /= np.linalg.norm(average_up)
        server.scene.set_up_direction((average_up[0], average_up[1], average_up[2]))

    @gui_reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.camera.up_direction = vtf.SO3(client.camera.wxyz) @ np.array(
            [0.0, -1.0, 0.0]
        )

    gui_points = server.gui.add_slider(
        "Max points",
        min=1,
        max=len(points3d),
        step=1,
        initial_value=min(len(points3d), 50_000),
    )
    gui_frames = server.gui.add_slider(
        "Max frames",
        min=1,
        max=len(images),
        step=1,
        initial_value=min(len(images), 50),
    )
    gui_point_size = server.gui.add_slider(
        "Point size", min=0.005, max=0.1, step=0.001, initial_value=0.02
    )
    gui_frustum_size = server.gui.add_slider(
        "Frustum size", min=0.01, max=0.1, step=0.001, initial_value=0.02
    )

    point_mask = np.random.choice(points.shape[0], gui_points.value, replace=False)
    point_cloud = server.scene.add_point_cloud(
        name="/colmap/pcd",
        points=points[point_mask],
        colors=colors[point_mask],
        point_size=gui_point_size.value,
        point_shape="circle",
    )
    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames() -> None:

        # Remove existing image frames.
        for frame in frames:
            frame.remove()
        frames.clear()

        # Interpret the images and cameras.
        img_ids = [im.id for im in images.values()]
        random.shuffle(img_ids)
        img_ids = sorted(img_ids[: gui_frames.value])

        for img_id in tqdm(img_ids):
            img = images[img_id]
            cam = cameras[img.camera_id]

            if args.images_path is not None:
                image_filename = args.images_path / img.name
                # Skip images that don't exist.
                if not image_filename.exists():
                    continue
                image = iio.imread(image_filename)
                image = image[::args.downsample_factor, ::args.downsample_factor]
            else:
                image = None

            T_world_camera = vtf.SE3.from_rotation_and_translation(
                vtf.SO3(img.qvec), img.tvec
            ).inverse()

            frame = server.scene.add_frame(
                f"/colmap/frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.1,
                axes_radius=0.005,
                show_axes=False,
            )
            frames.append(frame)

            # For pinhole cameras, cam.params will be (fx, fy, cx, cy).
            if cam.model != "PINHOLE":
                print(f"Expected pinhole camera, but got {cam.model}")

            H, W = cam.height, cam.width
            fy = cam.params[1]
            frustum = server.scene.add_camera_frustum(
                f"/colmap/frame_{img_id}/frustum",
                fov=2 * np.arctan2(H / 2, fy),
                aspect=W / H,
                scale=0.02,
                color=(255, 0, 0),
                image=image,
            )
            frustums.append(frustum)

            @frustum.on_click
            def _(_, frame=frame) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position
            
            if args.colmap_ref_path is None:
                continue

            img_ref = images_ref[img_id]
            cam_ref = cameras_ref[img_ref.camera_id]

            T_world_camera_ref = vtf.SE3.from_rotation_and_translation(
                vtf.SO3(img_ref.qvec), img_ref.tvec
            ).inverse()

            frame_ref = server.scene.add_frame(
                f"/colmap_ref/frame_{img_id}",
                wxyz=T_world_camera_ref.rotation().wxyz,
                position=T_world_camera_ref.translation(),
                axes_length=0.1,
                axes_radius=0.005,
                show_axes=False,
            )
            frames.append(frame_ref)

            # For pinhole cameras, cam.params will be (fx, fy, cx, cy).
            if cam_ref.model != "PINHOLE":
                print(f"Expected pinhole camera, but got {cam_ref.model}")

            H, W = cam_ref.height, cam_ref.width
            fy = cam_ref.params[1]
            frustum_ref = server.scene.add_camera_frustum(
                f"/colmap_ref/frame_{img_id}/frustum",
                fov=2 * np.arctan2(H / 2, fy),
                aspect=W / H,
                scale=0.02,
                color=(0, 255, 255),
            )
            frustums.append(frustum_ref)

            @frustum_ref.on_click
            def _(_, frame_ref=frame_ref) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame_ref.wxyz
                    client.camera.position = frame_ref.position

    need_update = True

    @gui_points.on_update
    def _(_) -> None:
        point_mask = np.random.choice(points.shape[0], gui_points.value, replace=False)
        with server.atomic():
            point_cloud.points = points[point_mask]
            point_cloud.colors = colors[point_mask]

    @gui_frames.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    @gui_point_size.on_update
    def _(_) -> None:
        point_cloud.point_size = gui_point_size.value
    
    @gui_frustum_size.on_update
    def _(_) -> None:
        for frustum in frustums:
            frustum.scale = gui_frustum_size.value

    while True:
        if need_update:
            need_update = False
            visualize_frames()

        time.sleep(1e-3)


if __name__ == "__main__":
    main()