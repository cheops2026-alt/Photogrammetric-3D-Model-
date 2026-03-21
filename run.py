# Photogrammetric-3D-Model
import argparse
import logging
import os
import time

import numpy as np
import rembg
import torch
import xatlas
from PIL import Image, ImageEnhance

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video
from tsr.bake_texture import bake_texture
from model_utils import show_viewer


def enhance_image(img: Image.Image) -> Image.Image:
    """Improve contrast, sharpness, and color for better reconstruction and texture."""
    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    img = ImageEnhance.Color(img).enhance(1.2)
    return img


def pick_images_via_dialog() -> list[str]:
    """Open a native file dialog to choose one or more images."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        paths = filedialog.askopenfilenames(
            title="Select image(s) for 3D reconstruction",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.webp *.bmp"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return list(paths) if paths else []
    except Exception as exc:
        logging.warning(f"File dialog unavailable: {exc}")
        return []


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
    help="Path to input image(s). Omit to use examples/chair.png if present.",
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
    default=384,
    type=int,
    help="Marching cubes grid resolution (higher = more detail). Default: 384",
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
    help="Bake a texture atlas for realistic colors (default: on)",
)
parser.add_argument(
    "--no-bake-texture",
    action="store_true",
    help="Use vertex colors only (faster, less detailed color)",
)
parser.add_argument(
    "--texture-resolution",
    default=4096,
    type=int,
    help="Texture atlas resolution (higher = sharper). Default: 4096",
)
parser.add_argument(
    "--save-to",
    type=str,
    default=None,
    help="Folder used when saving from the 3D viewer (S / G / checkbox). Default: Desktop/3DModel_Output",
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

# No paths on CLI → open file picker, then optional example fallback
if not args.image:
    picked = pick_images_via_dialog()
    if picked:
        args.image = picked
        logging.info(f"Selected {len(picked)} image(s) from file dialog.")
    else:
        _here = os.path.dirname(os.path.abspath(__file__))
        _default_img = os.path.join(_here, "examples", "chair.png")
        if os.path.isfile(_default_img):
            args.image = [_default_img]
            logging.info(f"No image selected; using default example: {_default_img}")
        else:
            parser.error(
                "No input image. Choose a file in the dialog, or run e.g.:\n"
                "  python run.py path/to/photo.png\n"
                f"Or add examples/chair.png under {_here}"
            )

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
first_mesh_path = None
first_texture_path = None
save_dest = (
    args.save_to
    if args.save_to
    else os.path.join(os.path.expanduser("~"), "Desktop", "3DModel_Output")
)

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

for i, image_path in enumerate(args.image):
    if args.no_remove_bg:
        image = np.array(Image.open(image_path).convert("RGB"))
    else:
        pil_in = Image.open(image_path).convert("RGB")
        pil_in = enhance_image(pil_in)
        image = remove_background(pil_in, rembg_session)
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
        if first_texture_path is None:
            first_texture_path = out_texture_path

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

# 3D viewer: textured preview + Save GLB checkbox / G key
if not args.no_viewer and first_mesh_path and os.path.exists(first_mesh_path):
    logging.info("Opening 3D viewer...")
    input_png = os.path.join(os.path.dirname(os.path.abspath(first_mesh_path)), "input.png")
    try:
        show_viewer(
            mesh_path=first_mesh_path,
            texture_path=first_texture_path,
            save_dest=save_dest,
            input_photo_path=input_png if os.path.exists(input_png) else None,
            title_prefix="TripoSR",
        )
    except Exception as e:
        logging.warning(f"Python viewer failed: {e}. Opening with system default...")
        os.startfile(os.path.abspath(first_mesh_path))
