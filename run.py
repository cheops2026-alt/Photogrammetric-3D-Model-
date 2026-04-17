# TripoSR pipeline: turn one or more photos into a 3D mesh (and optional texture).
# Rough order of phases:
#   1) Pick input images (CLI, file dialog, or default example).
#   2) Load the TripoSR model on GPU or CPU.
#   3) For each image: clean the photo, run the model, extract mesh, bake/export.
#   4) Optionally refine geometry with depth, multi-view fusion, or an AI API.
#   5) Optionally open a 3D viewer to inspect and save results.
import argparse
import logging
import os
import time
import numpy as np
import rembg
import torch
import xatlas
from PIL import Image, ImageEnhance, ImageFilter
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video
from tsr.bake_texture import bake_texture
from model_utils import refine_mesh_with_ai, show_viewer
from depth_enhance import (
    enhance_mesh_with_depth,
    multi_view_depth_fusion,
)

def enhance_image(img: Image.Image) -> Image.Image:
    """Make the input photo a bit stronger before the model sees it (contrast, sharpness, color)."""
    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    img = ImageEnhance.Color(img).enhance(1.2)
    return img

def enhance_texture(tex: Image.Image) -> Image.Image:
    """Polish the baked texture image after it is created (sharper, richer colors)."""
    tex = ImageEnhance.Sharpness(tex).enhance(1.6)
    tex = ImageEnhance.Color(tex).enhance(1.15)
    tex = ImageEnhance.Contrast(tex).enhance(1.08)
    tex = tex.filter(ImageFilter.SMOOTH_MORE)
    tex = ImageEnhance.Sharpness(tex).enhance(1.3)
    return tex


def pick_images_via_dialog() -> list[str]:
    """Let the user pick image files with a normal desktop file window (Windows/macOS/Linux)."""
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
    """Simple stopwatch: logs when a step starts and how long it took (uses milliseconds)."""

    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0  # convert seconds to milliseconds for display
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

# --- Logging: print messages with time and level to the console ---
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
# --- Phase 0: read command-line flags (paths, quality, optional features) ---
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
parser.add_argument(
    "--refine-api-url",
    type=str,
    default=None,
    help="URL of an online AI service for back-side completion / mesh refinement.",
)
parser.add_argument(
    "--refine-api-key",
    type=str,
    default=None,
    help="Bearer token / API key for the refinement service.",
)
parser.add_argument(
    "--depth-enhance",
    action="store_true",
    default=True,
    help="Use MiDaS depth estimation to refine mesh geometry (default: on).",
)
parser.add_argument(
    "--no-depth-enhance",
    action="store_true",
    help="Skip MiDaS depth-based mesh refinement.",
)
parser.add_argument(
    "--smooth-iterations",
    type=int,
    default=2,
    help="Laplacian smoothing passes during mesh refinement. Default: 2",
)
parser.add_argument(
    "--multi-view",
    action="store_true",
    help="When multiple images are given, also run multi-view depth fusion.",
)
args = parser.parse_args()

# Turn "negative" flags into simple on/off settings the rest of the script can read.
if args.no_bake_texture:
    args.bake_texture = False
if args.no_depth_enhance:
    args.depth_enhance = False

# --- Phase 1: decide which image file(s) to use ---
# If the user did not pass paths on the command line: open a file picker, then fall back to a built-in example.
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

# Where to write meshes, textures, and debug images (subfolders per image index).
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
# Remember the first mesh/texture so later optional steps and the viewer know what to open.
first_mesh_path = None
first_texture_path = None
# Folder used when the user saves from the 3D viewer (hotkeys / UI).
save_dest = (
    args.save_to
    if args.save_to
    else os.path.join(os.path.expanduser("~"), "Desktop", "3DModel_Output")
)

# Use CUDA if available; otherwise run on CPU (slower but works without a GPU).
device = args.device
if not torch.cuda.is_available():
    device = "cpu"

# --- Phase 2: load TripoSR weights and move the model to the chosen device ---
timer.start("Initializing model")
model = TSR.from_pretrained(
    args.pretrained_model_name_or_path,
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(args.chunk_size)
model.to(device)
timer.end("Initializing model")

# --- Phase 3: prepare each input image (background removal + crop, or use as-is) ---
timer.start("Processing images")
images = []

if args.no_remove_bg:
    rembg_session = None
else:
    # One shared session speeds up repeated background removal.
    rembg_session = rembg.new_session()

for i, image_path in enumerate(args.image):
    if args.no_remove_bg:
        # User already prepared an RGB image on gray background; load it directly.
        image = np.array(Image.open(image_path).convert("RGB"))
    else:
        # Typical path: enhance → cut out subject → place on gray → save a copy as input.png.
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

# --- Phase 4: for each prepared image, run inference and export mesh (and texture if enabled) ---
for i, image in enumerate(images):
    logging.info(f"Running image {i + 1}/{len(images)} ...")

    # Encode the image into an internal scene representation (the neural forward pass).
    timer.start("Running model")
    with torch.no_grad():
        scene_codes = model([image], device=device)
    timer.end("Running model")

    # Optional: save many rendered views and a short video (helps debug or presentations).
    if args.render:
        timer.start("Rendering")
        render_images = model.render(scene_codes, n_views=30, return_type="pil")
        for ri, render_image in enumerate(render_images[0]):
            render_image.save(os.path.join(output_dir, str(i), f"render_{ri:03d}.png"))
        save_video(
            render_images[0], os.path.join(output_dir, str(i), f"render.mp4"), fps=30
        )
        timer.end("Rendering")

    # Turn the learned field into a triangle mesh (marching cubes at mc_resolution).
    timer.start("Extracting mesh")
    meshes = model.extract_mesh(scene_codes, not args.bake_texture, resolution=args.mc_resolution)
    timer.end("Extracting mesh")

    # Paths for this image index (e.g. output/0/mesh.obj).
    out_mesh_path = os.path.join(output_dir, str(i), f"mesh.{args.model_save_format}")
    if first_mesh_path is None:
        first_mesh_path = out_mesh_path
    if args.bake_texture:
        out_texture_path = os.path.join(output_dir, str(i), "texture.png")
        if first_texture_path is None:
            first_texture_path = out_texture_path

        # Unwrap the mesh to a 2D texture atlas and sample colors from the model.
        timer.start("Baking texture")
        bake_output = bake_texture(meshes[0], model, scene_codes[0], args.texture_resolution)
        timer.end("Baking texture")

        # Write OBJ (or GLB path elsewhere) with UVs plus the PNG texture file.
        timer.start("Exporting mesh and texture")
        xatlas.export(
            out_mesh_path,
            meshes[0].vertices[bake_output["vmapping"]],
            bake_output["indices"],
            bake_output["uvs"],
            meshes[0].vertex_normals[bake_output["vmapping"]],
        )
        tex_img = Image.fromarray(
            (bake_output["colors"] * 255.0).astype(np.uint8)
        ).transpose(Image.FLIP_TOP_BOTTOM)
        tex_img = enhance_texture(tex_img)
        tex_img.save(out_texture_path, quality=95)
        timer.end("Exporting mesh and texture")
    else:
        # Vertex colors only: faster export, no separate texture file.
        timer.start("Exporting mesh")
        meshes[0].export(out_mesh_path)
        timer.end("Exporting mesh")

# --- Phase 5 (optional): use depth from the photo to improve mesh shape (MiDaS) ---
# Only runs on the first output mesh; needs input.png next to that mesh.
if args.depth_enhance and first_mesh_path and os.path.exists(first_mesh_path):
    input_img_for_depth = os.path.join(os.path.dirname(first_mesh_path), "input.png")
    if os.path.exists(input_img_for_depth):
        timer.start("Depth-based mesh enhancement")
        try:
            refined_path, _ = enhance_mesh_with_depth(
                mesh_path=first_mesh_path,
                texture_path=first_texture_path,
                input_image_path=input_img_for_depth,
                output_dir=os.path.dirname(first_mesh_path),
                device=device,
                smooth_iterations=args.smooth_iterations,
            )
            first_mesh_path = refined_path
        except Exception as exc:
            logging.warning(f"Depth enhancement failed: {exc}. Using original mesh.")
        timer.end("Depth-based mesh enhancement")

# --- Phase 6 (optional): combine depth from several photos into one extra mesh ---
if args.multi_view and len(args.image) > 1:
    timer.start("Multi-view depth fusion")
    try:
        mv_dir = os.path.join(output_dir, "multiview")
        os.makedirs(mv_dir, exist_ok=True)
        mv_mesh_path = multi_view_depth_fusion(
            image_paths=args.image,
            output_dir=mv_dir,
            device=device,
        )
        logging.info(f"Multi-view mesh available at: {mv_mesh_path}")
    except Exception as exc:
        logging.warning(f"Multi-view fusion failed: {exc}")
    timer.end("Multi-view depth fusion")

# --- Phase 7 (optional): send mesh to a remote AI service for extra cleanup or back-side detail ---
if first_mesh_path and os.path.exists(first_mesh_path) and args.refine_api_url:
    input_img_for_refine = os.path.join(os.path.dirname(first_mesh_path), "input.png")
    first_mesh_path, first_texture_path = refine_mesh_with_ai(
        mesh_path=first_mesh_path,
        texture_path=first_texture_path,
        input_image_path=input_img_for_refine if os.path.exists(input_img_for_refine) else None,
        output_dir=os.path.dirname(first_mesh_path),
        api_url=args.refine_api_url,
        api_key=args.refine_api_key,
    )

# --- Phase 8 (optional): open an interactive 3D preview; user can save GLB from there ---
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
        # On Windows, fall back to whatever app is registered for .obj/.glb files.
        logging.warning(f"Python viewer failed: {e}. Opening with system default...")
        os.startfile(os.path.abspath(first_mesh_path))
