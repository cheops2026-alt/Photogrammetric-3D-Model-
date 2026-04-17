# Photogrammetric-3D-Model - ESP32-CAM capture & 3D reconstruction
# ESP32-CAM URL: http://172.20.10.4
import argparse
import logging
import os
import time
from io import BytesIO

import cv2
import numpy as np
import requests
import rembg
import torch
import xatlas
from PIL import Image, ImageEnhance

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video
from tsr.bake_texture import bake_texture
from model_utils import refine_mesh_with_ai, export_glb, show_viewer


ESP32_BASE_URL   = "http://172.20.10.4"
ESP32_STREAM_URL = f"{ESP32_BASE_URL}:81/stream"   # MJPEG stream
ESP32_SNAP_URL   = f"{ESP32_BASE_URL}/capture"     # single JPEG snapshot


def enhance_image(img: Image.Image) -> Image.Image:
    """Sharpen, improve contrast and colour saturation."""
    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    img = ImageEnhance.Color(img).enhance(1.2)
    return img


def fetch_esp32_snapshot(snap_url: str) -> np.ndarray | None:
    """Grab a single JPEG frame from the ESP32-CAM /capture endpoint."""
    try:
        resp = requests.get(snap_url, timeout=5)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        logging.warning(f"Snapshot fetch failed: {exc}")
        return None


def capture_photo_esp32(
    output_path: str = "captured.png",
    stream_url: str = ESP32_STREAM_URL,
    snap_url: str   = ESP32_SNAP_URL,
) -> str | None:
    """
    Stream live video from ESP32-CAM and let the user capture a photo.

    The function tries the MJPEG stream first (real-time preview).
    If the stream is unavailable it falls back to polling /capture.

    Controls:
      SPACE / S  – start 3-second countdown then capture
      Q / ESC    – cancel
    """
    logging.info(f"Connecting to ESP32-CAM stream: {stream_url}")
    logging.info("Press SPACE or 'S' to capture  |  Q or ESC to quit")

    # Try MJPEG stream first
    stream = None
    stream_iter = None
    try:
        stream = requests.get(stream_url, stream=True, timeout=5)
        if stream.status_code != 200:
            stream = None
        else:
            # IMPORTANT: iter_content() can only be consumed once; keep a single iterator.
            stream_iter = stream.iter_content(chunk_size=4096)
    except Exception:
        stream = None
        stream_iter = None

    if stream:
        logging.info("MJPEG stream connected.")
    else:
        logging.warning("MJPEG stream unavailable – falling back to snapshot polling.")

    captured_path  = None
    countdown_start = None
    COUNTDOWN_SECS  = 3
    mjpeg_buf       = bytes()
    window_name     = "ESP32-CAM - TripoSR Capture"

    def next_frame_from_stream(it) -> np.ndarray | None:
        """Parse one JPEG frame from the MJPEG byte stream."""
        nonlocal mjpeg_buf
        try:
            for chunk in it:
                if not chunk:
                    continue
                mjpeg_buf += chunk
                start = mjpeg_buf.find(b"\xff\xd8")  # JPEG SOI marker
                end   = mjpeg_buf.find(b"\xff\xd9")  # JPEG EOI marker
                if start != -1 and end != -1 and end > start:
                    jpg = mjpeg_buf[start : end + 2]
                    mjpeg_buf = mjpeg_buf[end + 2 :]
                    arr = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    return frame
        except requests.exceptions.StreamConsumedError:
            # Will be handled by reconnect logic in the main loop.
            return None
        except Exception:
            return None
        return None

    while True:
        # --- Get frame ---
        if stream and stream_iter is not None:
            frame = next_frame_from_stream(stream_iter)
            if frame is None:
                # Reconnect stream (handles StreamConsumedError / dropouts)
                try:
                    stream.close()
                except Exception:
                    pass
                stream = None
                stream_iter = None
                try:
                    stream = requests.get(stream_url, stream=True, timeout=5)
                    if stream.status_code == 200:
                        stream_iter = stream.iter_content(chunk_size=4096)
                        logging.info("MJPEG stream reconnected.")
                    else:
                        stream = None
                        stream_iter = None
                except Exception:
                    stream = None
                    stream_iter = None
        else:
            frame = fetch_esp32_snapshot(snap_url)
            time.sleep(0.05)          # ~20 fps polling limit

        if frame is None:
            logging.error("No frame received from ESP32-CAM. Check the IP and connection.")
            cv2.waitKey(500)
            continue

        display = frame.copy()
        h, w = display.shape[:2]

        # Centred guide rectangle
        rx1, ry1 = int(w * 0.25), int(h * 0.10)
        rx2, ry2 = int(w * 0.75), int(h * 0.90)
        cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (0, 220, 0), 2)

        # ESP32-CAM label
        cv2.putText(display, "ESP32-CAM", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        if countdown_start is not None:
            elapsed   = time.time() - countdown_start
            remaining = COUNTDOWN_SECS - int(elapsed)
            if remaining > 0:
                cv2.putText(display, str(remaining),
                            (w // 2 - 30, h // 2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 60, 255), 8)
                cv2.putText(display, "Hold still!",
                            (w // 2 - 80, h // 2 + 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
            else:
                # Grab a fresh high-quality snapshot for the actual capture
                snap = fetch_esp32_snapshot(snap_url)
                raw  = snap if snap is not None else frame
                img  = Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
                img  = enhance_image(img)
                img.save(output_path, quality=95)
                captured_path = output_path
                logging.info(f"Photo captured and saved to {output_path}")

                # Flash effect
                flash = np.ones_like(display) * 255
                cv2.imshow(window_name, flash)
                cv2.waitKey(150)
                break
        else:
            cv2.putText(display, "CENTER OBJECT IN BOX",
                        (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)
            cv2.putText(display, "SPACE/S: Capture  |  Q/ESC: Quit",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            logging.info("Capture cancelled.")
            break
        elif key in (ord(" "), ord("s")) and countdown_start is None:
            countdown_start = time.time()

    if stream:
        stream.close()
    cv2.destroyAllWindows()
    return captured_path


# ===================================================================
# Timer
# ===================================================================
class Timer:
    def __init__(self):
        self.items     = {}
        self.time_scale = 1000.0
        self.time_unit  = "ms"

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
        t = (time.time() - start_time) * self.time_scale
        logging.info(f"{name} finished in {t:.2f}{self.time_unit}.")


timer = Timer()

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

# ===================================================================
# Arguments
# ===================================================================
parser = argparse.ArgumentParser(
    description="TripoSR 3D reconstruction from ESP32-CAM or image files."
)
parser.add_argument("image", type=str, nargs="*",
                    help="Path to input image(s). Omit to capture from ESP32-CAM.")
parser.add_argument("--capture", action="store_true",
                    help="Explicitly use ESP32-CAM capture mode.")
parser.add_argument("--esp-url", type=str, default=ESP32_BASE_URL,
                    help=f"Base URL of the ESP32-CAM. Default: {ESP32_BASE_URL}")
parser.add_argument("--device", default="cuda:0", type=str,
                    help="PyTorch device. Falls back to 'cpu' if CUDA unavailable.")
parser.add_argument("--pretrained-model-name-or-path",
                    default="stabilityai/TripoSR", type=str)
parser.add_argument("--chunk-size", default=8192, type=int,
                    help="Chunk size for surface extraction. Default: 8192")
parser.add_argument("--mc-resolution", default=384, type=int,
                    help="Marching cubes resolution. Higher = more detail. Default: 384")
parser.add_argument("--no-remove-bg", action="store_true",
                    help="Skip background removal.")
parser.add_argument("--foreground-ratio", default=0.85, type=float,
                    help="Foreground size ratio. Default: 0.85")
parser.add_argument("--output-dir", default="output/", type=str,
                    help="Output directory. Default: 'output/'")
parser.add_argument("--model-save-format", default="obj", type=str,
                    choices=["obj", "glb"], help="Mesh save format. Default: 'obj'")
parser.add_argument("--bake-texture", action="store_true", default=True,
                    help="Bake texture atlas for realistic colours. Default: True")
parser.add_argument("--no-bake-texture", action="store_true",
                    help="Use vertex colours instead (faster, less accurate).")
parser.add_argument("--texture-resolution", default=4096, type=int,
                    help="Texture atlas resolution. Default: 4096")
parser.add_argument("--render", action="store_true",
                    help="Save a NeRF-rendered 360 video.")
parser.add_argument("--no-viewer", action="store_true",
                    help="Skip opening 3D viewer after generation.")
parser.add_argument("--save-to", type=str, default=None,
                    help="Copy output files here. Defaults to Desktop/3DModel_Output.")
parser.add_argument("--refine-api-url", type=str, default=None,
                    help="URL of an online AI service for back-side completion and mesh refinement.")
parser.add_argument("--refine-api-key", type=str, default=None,
                    help="Bearer token / API key for the refinement service.")
args = parser.parse_args()

if args.no_bake_texture:
    args.bake_texture = False

# Derive ESP32 URLs from --esp-url
esp_stream = f"{args.esp_url}:81/stream"
esp_snap   = f"{args.esp_url}/capture"

# ===================================================================
# Input: ESP32-CAM or file
# ===================================================================
image_paths = list(args.image)
if args.capture or not image_paths:
    os.makedirs(args.output_dir, exist_ok=True)
    captured_path = os.path.join(args.output_dir, "captured.png")
    path = capture_photo_esp32(
        output_path=captured_path,
        stream_url=esp_stream,
        snap_url=esp_snap,
    )
    if path:
        image_paths = [path]
    else:
        logging.error("No photo captured. Exiting.")
        exit(1)

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)
first_mesh_path    = None
first_texture_path = None

device = args.device
if not torch.cuda.is_available():
    device = "cpu"

# ===================================================================
# Model init
# ===================================================================
timer.start("Initializing model")
model = TSR.from_pretrained(
    args.pretrained_model_name_or_path,
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(args.chunk_size)
model.to(device)
timer.end("Initializing model")

# ===================================================================
# Image preprocessing
# ===================================================================
timer.start("Processing images")
images = []
rembg_session = None if args.no_remove_bg else rembg.new_session()

for i, image_path in enumerate(image_paths):
    if args.no_remove_bg:
        image = np.array(Image.open(image_path).convert("RGB"))
    else:
        pil_img = Image.open(image_path)
        pil_img = enhance_image(pil_img.convert("RGB"))
        image   = remove_background(pil_img, rembg_session)
        image   = resize_foreground(image, args.foreground_ratio)
        image   = np.array(image).astype(np.float32) / 255.0
        image   = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image   = Image.fromarray((image * 255.0).astype(np.uint8))
        os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)
        image.save(os.path.join(output_dir, str(i), "input.png"))
    images.append(image)
timer.end("Processing images")

# ===================================================================
# 3D reconstruction
# ===================================================================
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
        tex_img = Image.fromarray(
            (bake_output["colors"] * 255.0).astype(np.uint8)
        ).transpose(Image.FLIP_TOP_BOTTOM)
        tex_img.save(out_texture_path)
        timer.end("Exporting mesh and texture")
    else:
        timer.start("Exporting mesh")
        meshes[0].export(out_mesh_path)
        timer.end("Exporting mesh")

# ===================================================================
# AI Refinement (optional)
# ===================================================================
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

# ===================================================================
# Save destination (used by viewer buttons, nothing is copied automatically)
# ===================================================================
save_dest = args.save_to if args.save_to else os.path.join(
    os.path.expanduser("~"), "Desktop", "3DModel_Output"
)

# ===================================================================
# 3D Viewer
# ===================================================================
if not args.no_viewer and first_mesh_path and os.path.exists(first_mesh_path):
    logging.info("Opening 3D viewer...")
    input_png = os.path.join(os.path.dirname(os.path.abspath(first_mesh_path)), "input.png")
    try:
        show_viewer(
            mesh_path=first_mesh_path,
            texture_path=first_texture_path,
            save_dest=save_dest,
            input_photo_path=input_png if os.path.exists(input_png) else None,
            title_prefix="TripoSR ESP32-CAM",
        )
    except Exception as e:
        logging.warning(f"Python viewer failed: {e}. Opening with system default...")
        os.startfile(os.path.abspath(first_mesh_path))
