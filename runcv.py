# Photogrammetric-3D-Model - Webcam capture & 3D reconstruction
import argparse
import logging
import os
import time

import cv2
import numpy as np
import rembg
import torch
import xatlas
from PIL import Image, ImageEnhance, ImageFilter

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video
from tsr.bake_texture import bake_texture


def enhance_image(img: Image.Image) -> Image.Image:
    """Sharpen, improve contrast and color saturation for better 3D reconstruction."""
    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    img = ImageEnhance.Color(img).enhance(1.2)
    return img


def capture_photo(output_path="captured.png", camera_id=0, cam_width=1280, cam_height=720):
    """Capture a HD photo from webcam. Press SPACE or 's' to capture, 'q'/ESC to quit."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Make sure a camera is connected.")

    # Request HD resolution from camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logging.info(f"Camera opened at {actual_w}x{actual_h}")
    logging.info("Press SPACE or 'S' to capture | 'Q' or ESC to quit")

    captured_path = None
    countdown_start = None
    COUNTDOWN_SECS = 3

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to read from camera.")
            break

        frame = cv2.flip(frame, 1)  # mirror for natural view
        display = frame.copy()
        h, w = display.shape[:2]

        # Draw a centered guide rectangle (~50% of frame)
        rx1, ry1 = int(w * 0.25), int(h * 0.1)
        rx2, ry2 = int(w * 0.75), int(h * 0.9)
        cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (0, 220, 0), 2)

        if countdown_start is not None:
            elapsed = time.time() - countdown_start
            remaining = COUNTDOWN_SECS - int(elapsed)
            if remaining > 0:
                cv2.putText(display, str(remaining),
                            (w // 2 - 30, h // 2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 60, 255), 8)
                cv2.putText(display, "Hold still!",
                            (w // 2 - 80, h // 2 + 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
            else:
                # Capture now
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = enhance_image(img)
                img.save(output_path, quality=95)
                captured_path = output_path
                logging.info(f"Photo captured and saved to {output_path}")

                # Flash effect
                flash = np.ones_like(display) * 255
                cv2.imshow("TripoSR - Capture Photo", flash)
                cv2.waitKey(150)
                break
        else:
            cv2.putText(display, "CENTER OBJECT IN BOX",
                        (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)
            cv2.putText(display, "SPACE/S: Capture  |  Q/ESC: Quit",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("TripoSR - Capture Photo", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            logging.info("Capture cancelled.")
            break
        elif key in (ord(" "), ord("s")) and countdown_start is None:
            countdown_start = time.time()

    cap.release()
    cv2.destroyAllWindows()
    return captured_path


class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0
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
    help="Path to input image(s). Omit to capture from webcam.",
)
parser.add_argument("--capture", action="store_true",
                    help="Explicitly use webcam capture mode.")
parser.add_argument("--camera", type=int, default=0,
                    help="Camera device ID. Default: 0")
parser.add_argument("--cam-width", type=int, default=1280,
                    help="Webcam capture width. Default: 1280")
parser.add_argument("--cam-height", type=int, default=720,
                    help="Webcam capture height. Default: 720")
parser.add_argument("--device", default="cuda:0", type=str,
                    help="Device to use. Falls back to 'cpu' if CUDA unavailable.")
parser.add_argument("--pretrained-model-name-or-path",
                    default="stabilityai/TripoSR", type=str)
parser.add_argument("--chunk-size", default=8192, type=int,
                    help="Chunk size for surface extraction. Default: 8192")
parser.add_argument("--mc-resolution", default=320, type=int,
                    help="Marching cubes resolution. Higher = more detail. Default: 320")
parser.add_argument("--no-remove-bg", action="store_true",
                    help="Skip background removal (image must already have gray bg).")
parser.add_argument("--foreground-ratio", default=0.85, type=float,
                    help="Foreground size ratio. Default: 0.85")
parser.add_argument("--output-dir", default="output/", type=str,
                    help="Output directory. Default: 'output/'")
parser.add_argument("--model-save-format", default="obj", type=str,
                    choices=["obj", "glb"],
                    help="Mesh save format. Default: 'obj'")
parser.add_argument("--bake-texture", action="store_true", default=True,
                    help="Bake texture atlas for realistic colors. Default: True")
parser.add_argument("--no-bake-texture", action="store_true",
                    help="Use vertex colors instead (faster, less accurate).")
parser.add_argument("--texture-resolution", default=2048, type=int,
                    help="Texture atlas resolution. Default: 2048")
parser.add_argument("--render", action="store_true",
                    help="Save a NeRF-rendered 360 video.")
parser.add_argument("--no-viewer", action="store_true",
                    help="Skip opening 3D viewer after generation.")
args = parser.parse_args()

if args.no_bake_texture:
    args.bake_texture = False

# --- Input: webcam or file ---
image_paths = list(args.image)
if args.capture or not image_paths:
    os.makedirs(args.output_dir, exist_ok=True)
    captured_path = os.path.join(args.output_dir, "captured.png")
    path = capture_photo(
        output_path=captured_path,
        camera_id=args.camera,
        cam_width=args.cam_width,
        cam_height=args.cam_height,
    )
    if path:
        image_paths = [path]
    else:
        logging.error("No photo captured. Exiting.")
        exit(1)

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
first_mesh_path = None
first_texture_path = None

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

rembg_session = None if args.no_remove_bg else rembg.new_session()

for i, image_path in enumerate(image_paths):
    if args.no_remove_bg:
        image = np.array(Image.open(image_path).convert("RGB"))
    else:
        pil_img = Image.open(image_path)
        # Enhance before background removal for better mask quality
        pil_img = enhance_image(pil_img.convert("RGB"))
        image = remove_background(pil_img, rembg_session)
        image = resize_foreground(image, args.foreground_ratio)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)
        image.save(os.path.join(output_dir, str(i), "input.png"))
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
        save_video(render_images[0], os.path.join(output_dir, str(i), "render.mp4"), fps=30)
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
        xatlas.export(
            out_mesh_path,
            meshes[0].vertices[bake_output["vmapping"]],
            bake_output["indices"],
            bake_output["uvs"],
            meshes[0].vertex_normals[bake_output["vmapping"]],
        )
        # Save texture with high quality
        tex_img = Image.fromarray(
            (bake_output["colors"] * 255.0).astype(np.uint8)
        ).transpose(Image.FLIP_TOP_BOTTOM)
        tex_img.save(out_texture_path)
        timer.end("Exporting mesh and texture")
    else:
        timer.start("Exporting mesh")
        meshes[0].export(out_mesh_path)
        timer.end("Exporting mesh")

# --- 3D Viewer ---
if not args.no_viewer and first_mesh_path and os.path.exists(first_mesh_path):
    logging.info("Opening 3D viewer...")
    try:
        os.environ["VTK_LOGGING_LEVEL"] = "ERROR"
        import pyvista as pv

        mesh_path = os.path.abspath(first_mesh_path)
        mesh = pv.read(mesh_path)

        # Side-by-side: input image + 3D model
        input_png = os.path.join(os.path.dirname(mesh_path), "input.png")
        has_input = os.path.exists(input_png)
        has_texture = first_texture_path and os.path.exists(first_texture_path)

        pl = pv.Plotter(
            shape=(1, 2) if has_input else (1, 1),
            window_size=[1200, 650] if has_input else [800, 650],
        )

        # Left panel: input photo
        if has_input:
            pl.subplot(0, 0)
            img_data = pv.read(input_png)
            pl.add_mesh(img_data, rgb=True)
            pl.add_title("Input Photo", font_size=12)
            pl.view_xy()
            pl.subplot(0, 1)

        # Right panel (or only panel): 3D mesh
        if has_texture:
            texture = pv.read_texture(first_texture_path)
            pl.add_mesh(mesh, texture=texture, show_edges=False, smooth_shading=True)
        else:
            pl.add_mesh(mesh, show_edges=False, smooth_shading=True,
                        color="lightgray", pbr=True, metallic=0.1, roughness=0.5)

        pl.add_light(pv.Light(position=(5, 5, 5), intensity=0.7))
        pl.add_light(pv.Light(position=(-5, -5, 5), intensity=0.4))
        pl.set_background("white")
        pl.add_title("TripoSR - 3D Output  (drag: rotate | scroll: zoom | R: reset)", font_size=10)
        pl.show(title="TripoSR - 3D Viewer")
    except Exception as e:
        logging.warning(f"Python viewer failed: {e}. Opening with system default...")
        os.startfile(os.path.abspath(first_mesh_path))
