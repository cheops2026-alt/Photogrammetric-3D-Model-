import argparse
import logging
import os
import time

import cv2
import numpy as np
import rembg
import torch
import xatlas
from PIL import Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video
from tsr.bake_texture import bake_texture


def capture_photo(output_path="captured.png", camera_id=0):
    """Capture a photo from webcam using OpenCV. Press SPACE or 's' to capture, 'q' to quit."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Make sure a camera is connected.")

    captured_path = None
    logging.info("Camera ready. Press SPACE or 's' to capture, 'q' to quit without capturing.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to read from camera.")
            break

        # Mirror the frame for a more natural view
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        # Add instructions on the frame
        cv2.putText(
            display, "SPACE or S: Capture | Q: Quit",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        cv2.imshow("TripoSR - Capture Photo", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:  # q or ESC
            logging.info("Capture cancelled.")
            break
        elif key == ord(" ") or key == ord("s"):  # SPACE or s
            # Convert BGR to RGB and save
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.save(output_path)
            captured_path = output_path
            logging.info(f"Photo captured and saved to {output_path}")
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_path


class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info(f"{name} ...")

    def end(self, name: str) -> float:
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        logging.info(f"{name} finished in {t:.2f}{self.time_unit}.")


timer = Timer()


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
parser = argparse.ArgumentParser()
parser.add_argument(
    "image",
    type=str,
    nargs="*",
    help="Path to input image(s). Omit when using --capture to take a photo from webcam.",
)
parser.add_argument(
    "--capture",
    action="store_true",
    help="Capture a photo from webcam before creating 3D model. Press SPACE to capture.",
)
parser.add_argument(
    "--camera",
    type=int,
    default=0,
    help="Camera device ID. Default: 0",
)
parser.add_argument(
    "--device",
    default="cuda:0",
    type=str,
    help="Device to use. If no CUDA-compatible device is found, will fallback to 'cpu'. Default: 'cuda:0'",
)
parser.add_argument(
    "--pretrained-model-name-or-path",
    default="stabilityai/TripoSR",
    type=str,
    help="Path to the pretrained model. Could be either a huggingface model id is or a local path. Default: 'stabilityai/TripoSR'",
)
parser.add_argument(
    "--chunk-size",
    default=8192,
    type=int,
    help="Evaluation chunk size for surface extraction and rendering. Smaller chunk size reduces VRAM usage but increases computation time. 0 for no chunking. Default: 8192",
)
parser.add_argument(
    "--mc-resolution",
    default=256,
    type=int,
    help="Marching cubes grid resolution. Default: 256"
)
parser.add_argument(
    "--no-remove-bg",
    action="store_true",
    help="If specified, the background will NOT be automatically removed from the input image, and the input image should be an RGB image with gray background and properly-sized foreground. Default: false",
)
parser.add_argument(
    "--foreground-ratio",
    default=0.85,
    type=float,
    help="Ratio of the foreground size to the image size. Only used when --no-remove-bg is not specified. Default: 0.85",
)
parser.add_argument(
    "--output-dir",
    default="output/",
    type=str,
    help="Output directory to save the results. Default: 'output/'",
)
parser.add_argument(
    "--model-save-format",
    default="obj",
    type=str,
    choices=["obj", "glb"],
    help="Format to save the extracted mesh. Default: 'obj'",
)
parser.add_argument(
    "--bake-texture",
    action="store_true",
    default=True,
    help="Bake a texture atlas for realistic colors (default: True)",
)
parser.add_argument(
    "--no-bake-texture",
    action="store_true",
    help="Use vertex colors instead of baked texture (faster but less accurate colors)",
)
parser.add_argument(
    "--texture-resolution",
    default=2048,
    type=int,
    help="Texture atlas resolution, only useful with --bake-texture. Default: 2048"
)
parser.add_argument(
    "--render",
    action="store_true",
    help="If specified, save a NeRF-rendered video. Default: false",
)
parser.add_argument(
    "--no-viewer",
    action="store_true",
    help="Do not open the output mesh after generation. Default: false",
)
args = parser.parse_args()
if args.no_bake_texture:
    args.bake_texture = False

# Determine input images: from --capture or from image paths
image_paths = list(args.image)
if args.capture or not image_paths:
    # No image provided - capture from webcam
    os.makedirs(args.output_dir, exist_ok=True)
    captured_path = os.path.join(args.output_dir, "captured.png")
    path = capture_photo(output_path=captured_path, camera_id=args.camera)
    if path:
        image_paths = [path]
    else:
        logging.error("No photo captured. Exiting.")
        exit(1)

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
first_mesh_path = None

device = args.device
if not torch.cuda.is_available():
    device = "cpu"

timer.start("Initializing model")
model = TSR.from_pretrained(
    args.pretrained_model_name_or_path,
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(args.chunk_size)
model.to(device)
timer.end("Initializing model")

timer.start("Processing images")
images = []

if args.no_remove_bg:
    rembg_session = None
else:
    rembg_session = rembg.new_session()

for i, image_path in enumerate(image_paths):
    if args.no_remove_bg:
        image = np.array(Image.open(image_path).convert("RGB"))
    else:
        image = remove_background(Image.open(image_path), rembg_session)
        image = resize_foreground(image, args.foreground_ratio)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        if not os.path.exists(os.path.join(output_dir, str(i))):
            os.makedirs(os.path.join(output_dir, str(i)))
        image.save(os.path.join(output_dir, str(i), f"input.png"))
    images.append(image)
timer.end("Processing images")

for i, image in enumerate(images):
    logging.info(f"Running image {i + 1}/{len(images)} ...")

    timer.start("Running model")
    with torch.no_grad():
        scene_codes = model([image], device=device)
    timer.end("Running model")

    if args.render:
        timer.start("Rendering")
        render_images = model.render(scene_codes, n_views=30, return_type="pil")
        for ri, render_image in enumerate(render_images[0]):
            render_image.save(os.path.join(output_dir, str(i), f"render_{ri:03d}.png"))
        save_video(
            render_images[0], os.path.join(output_dir, str(i), f"render.mp4"), fps=30
        )
        timer.end("Rendering")

    timer.start("Extracting mesh")
    meshes = model.extract_mesh(scene_codes, not args.bake_texture, resolution=args.mc_resolution)
    timer.end("Extracting mesh")

    out_mesh_path = os.path.join(output_dir, str(i), f"mesh.{args.model_save_format}")
    if first_mesh_path is None:
        first_mesh_path = out_mesh_path
    if args.bake_texture:
        out_texture_path = os.path.join(output_dir, str(i), "texture.png")

        timer.start("Baking texture")
        bake_output = bake_texture(meshes[0], model, scene_codes[0], args.texture_resolution)
        timer.end("Baking texture")

        timer.start("Exporting mesh and texture")
        xatlas.export(out_mesh_path, meshes[0].vertices[bake_output["vmapping"]], bake_output["indices"], bake_output["uvs"], meshes[0].vertex_normals[bake_output["vmapping"]])
        Image.fromarray((bake_output["colors"] * 255.0).astype(np.uint8)).transpose(Image.FLIP_TOP_BOTTOM).save(out_texture_path)
        timer.end("Exporting mesh and texture")
    else:
        timer.start("Exporting mesh")
        meshes[0].export(out_mesh_path)
        timer.end("Exporting mesh")

# Open 3D model in a Python viewer window
if not args.no_viewer and first_mesh_path and os.path.exists(first_mesh_path):
    logging.info("Opening 3D viewer...")
    try:
        os.environ["VTK_LOGGING_LEVEL"] = "ERROR"  # Reduce VTK log spam
        import pyvista as pv
        mesh_path = os.path.abspath(first_mesh_path)
        mesh = pv.read(mesh_path)
        pl = pv.Plotter(window_size=[800, 600])
        # Apply texture for colored output (when using --bake-texture)
        texture_path = os.path.join(os.path.dirname(mesh_path), "texture.png")
        if os.path.exists(texture_path):
            texture = pv.read_texture(texture_path)
            pl.add_mesh(mesh, texture=texture, show_edges=False)
        else:
            pl.add_mesh(mesh, show_edges=False)
        pl.show(title="TripoSR - 3D Output")
    except Exception as e:
        logging.warning(f"Python viewer failed: {e}. Opening with system default...")
        os.startfile(os.path.abspath(first_mesh_path))
