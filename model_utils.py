# Photogrammetric-3D-Model - Shared utilities for AI refinement, GLB export, and 3D viewer
import logging
import os

import trimesh
from PIL import Image


# ===================================================================
# AI Refinement (optional online stage)
# ===================================================================

def refine_mesh_with_ai(
    mesh_path: str,
    texture_path: str | None,
    input_image_path: str | None,
    output_dir: str,
    api_url: str | None = None,
    api_key: str | None = None,
) -> tuple[str, str | None]:
    """
    Send the initial mesh + input image to an online AI service for
    back-side completion and overall quality improvement.

    The service is expected to accept a POST with:
      - 'image': the original input photo (multipart file)
      - 'mesh':  the initial .obj or .glb file  (multipart file)
      - 'texture': the texture.png if present    (multipart file, optional)
    And return either:
      - A refined .glb binary (Content-Type: model/gltf-binary)
      - A JSON with a download URL for the refined asset

    Returns (refined_mesh_path, refined_texture_path).
    If the service returns a GLB, texture is embedded so refined_texture_path is None.
    Falls back to the original mesh on any error.
    """
    if not api_url:
        logging.info("AI refinement skipped (no --refine-api-url provided).")
        return mesh_path, texture_path

    try:
        import requests
    except ImportError:
        logging.warning("requests not installed; skipping AI refinement.")
        return mesh_path, texture_path

    logging.info(f"Sending mesh to AI refinement service: {api_url}")
    try:
        files = {}
        if input_image_path and os.path.exists(input_image_path):
            files["image"] = ("input.png", open(input_image_path, "rb"), "image/png")
        if os.path.exists(mesh_path):
            mime = "model/gltf-binary" if mesh_path.endswith(".glb") else "application/octet-stream"
            files["mesh"] = (os.path.basename(mesh_path), open(mesh_path, "rb"), mime)
        if texture_path and os.path.exists(texture_path):
            files["texture"] = ("texture.png", open(texture_path, "rb"), "image/png")

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        resp = requests.post(api_url, files=files, headers=headers, timeout=120)
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")

        if "gltf-binary" in content_type or "octet-stream" in content_type:
            refined_path = os.path.join(output_dir, "mesh_refined.glb")
            with open(refined_path, "wb") as f:
                f.write(resp.content)
            logging.info(f"AI refinement complete: {refined_path}")
            return refined_path, None

        # Try JSON with download URL
        data = resp.json()
        download_url = data.get("url") or data.get("download_url") or data.get("mesh_url")
        if download_url:
            dl = requests.get(download_url, timeout=60)
            dl.raise_for_status()
            refined_path = os.path.join(output_dir, "mesh_refined.glb")
            with open(refined_path, "wb") as f:
                f.write(dl.content)
            logging.info(f"AI refinement downloaded: {refined_path}")
            return refined_path, None

        logging.warning("AI service returned unexpected response; using original mesh.")
        return mesh_path, texture_path

    except Exception as exc:
        logging.warning(f"AI refinement failed: {exc}. Using original mesh.")
        return mesh_path, texture_path


# ===================================================================
# GLB Export
# ===================================================================

def export_glb(mesh_path: str, texture_path: str | None, glb_path: str) -> str:
    """
    Export the mesh (OBJ or GLB) + optional sidecar texture as a self-contained .glb.
    If the mesh is already a GLB without a sidecar texture, just copy it.
    Returns the path to the written .glb file.
    """
    if mesh_path.endswith(".glb") and texture_path is None:
        if os.path.abspath(mesh_path) != os.path.abspath(glb_path):
            import shutil
            shutil.copy2(mesh_path, glb_path)
        return glb_path

    scene = trimesh.load(mesh_path, process=False)
    mesh_obj = scene if isinstance(scene, trimesh.Trimesh) else list(scene.geometry.values())[0]

    if texture_path and os.path.exists(texture_path):
        tex_img = Image.open(texture_path)
        if mesh_obj.visual and hasattr(mesh_obj.visual, "uv") and mesh_obj.visual.uv is not None:
            from trimesh.visual.material import PBRMaterial
            material = PBRMaterial(baseColorTexture=tex_img)
            mesh_obj.visual = trimesh.visual.TextureVisuals(uv=mesh_obj.visual.uv, material=material)
        else:
            from trimesh.visual.material import PBRMaterial
            material = PBRMaterial(baseColorTexture=tex_img)
            mesh_obj.visual = trimesh.visual.TextureVisuals(material=material)

    glb_data = mesh_obj.export(file_type="glb")
    with open(glb_path, "wb") as f:
        f.write(glb_data)

    logging.info(f"GLB exported: {glb_path} ({os.path.getsize(glb_path) / 1024:.0f} KB)")
    return glb_path


# ===================================================================
# Enhanced 3D Viewer with GLB export action
# ===================================================================

def show_viewer(
    mesh_path: str,
    texture_path: str | None,
    save_dest: str,
    input_photo_path: str | None = None,
    title_prefix: str = "TripoSR",
):
    """
    Open a PyVista 3D viewer.

    Keyboard actions (nothing is saved automatically):
      S  – save OBJ mesh + texture + input photo to the output folder
      G  – export and save as a self-contained .glb
      O  – open the output folder in Explorer
    """
    import shutil
    os.environ["VTK_LOGGING_LEVEL"] = "ERROR"
    import pyvista as pv

    mesh = pv.read(os.path.abspath(mesh_path))

    has_input = input_photo_path and os.path.exists(input_photo_path)
    has_texture = texture_path and os.path.exists(texture_path)

    pl = pv.Plotter(
        shape=(1, 2) if has_input else (1, 1),
        window_size=[1400, 700] if has_input else [900, 700],
    )

    if has_input:
        pl.subplot(0, 0)
        img_data = pv.read(input_photo_path)
        pl.add_mesh(img_data, rgb=True)
        pl.add_title("Input Photo", font_size=9)
        pl.view_xy()
        pl.subplot(0, 1)

    if has_texture:
        texture = pv.read_texture(texture_path)
        pl.add_mesh(
            mesh, texture=texture, show_edges=False,
            smooth_shading=True, specular=0.3, specular_power=20,
        )
    else:
        pl.add_mesh(
            mesh, show_edges=False, smooth_shading=True,
            color="lightgray", pbr=True, metallic=0.05,
            roughness=0.4, specular=0.5,
        )

    pl.add_light(pv.Light(position=(5,  5,  8), intensity=0.8,  light_type="scene light"))
    pl.add_light(pv.Light(position=(-4, -3,  4), intensity=0.45, light_type="scene light"))
    pl.add_light(pv.Light(position=(0,  -6, -2), intensity=0.2,  light_type="scene light"))
    pl.set_background("white")
    pl.add_title("3D Output", font_size=9)
    pl.add_text(
        "Drag: rotate  Scroll: zoom  |  S: save files  G: save GLB  O: open folder",
        position="lower_edge", font_size=7, color="gray",
    )

    _mesh_path = os.path.abspath(mesh_path)
    _texture_path = os.path.abspath(texture_path) if has_texture else None
    _input_photo = os.path.abspath(input_photo_path) if has_input else None
    _save_dest = os.path.abspath(save_dest)

    def _ensure_dest():
        os.makedirs(_save_dest, exist_ok=True)

    def save_files():
        """S key: copy mesh, texture, and input photo to the output folder."""
        try:
            _ensure_dest()
            shutil.copy2(_mesh_path, _save_dest)
            print(f"\n  Mesh saved: {os.path.join(_save_dest, os.path.basename(_mesh_path))}")
            if _texture_path:
                shutil.copy2(_texture_path, _save_dest)
                print(f"  Texture saved: {os.path.join(_save_dest, os.path.basename(_texture_path))}")
            if _input_photo:
                shutil.copy2(_input_photo, _save_dest)
                print(f"  Input photo saved: {os.path.join(_save_dest, os.path.basename(_input_photo))}")
            print(f"  Folder: {_save_dest}\n")
        except Exception as exc:
            logging.warning(f"Save failed: {exc}")

    def save_glb():
        """G key: export a self-contained GLB into the output folder."""
        try:
            _ensure_dest()
            glb_out = os.path.join(_save_dest, "model.glb")
            export_glb(_mesh_path, _texture_path, glb_out)
            print(f"\n  GLB saved: {glb_out}\n")
        except Exception as exc:
            logging.warning(f"GLB export failed: {exc}")

    # Clickable checkbox (top-left) — same as G / download GLB
    def _on_glb_checkbox(state: bool) -> None:
        if state:
            save_glb()

    try:
        pl.add_checkbox_button_widget(
            callback=_on_glb_checkbox,
            value=False,
            position=(10.0, 10.0),
            size=44,
            border_size=3,
            color_on="seagreen",
            color_off="lightgray",
            background_color="white",
        )
        pl.add_text(
            "Save GLB",
            position=(62.0, 18.0),
            font_size=9,
            color="dimgray",
            viewport=True,
        )
    except Exception as exc:
        logging.debug(f"Checkbox widget not added: {exc}")

    def open_save_folder():
        """O key: open the output folder in Explorer."""
        if os.path.exists(_save_dest):
            os.startfile(_save_dest)
        else:
            print(f"\n  Nothing saved yet. Press S or G first.\n")

    pl.add_key_event("s", save_files)
    pl.add_key_event("S", save_files)
    pl.add_key_event("g", save_glb)
    pl.add_key_event("G", save_glb)
    pl.add_key_event("o", open_save_folder)
    pl.add_key_event("O", open_save_folder)
    pl.reset_camera()
    pl.show(title=f"{title_prefix} - 3D Viewer")
