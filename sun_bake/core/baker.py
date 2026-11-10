import os
import re
import json
import time
import shutil
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any
import numpy as np
import cv2
from psd_tools import PSDImage

from .model import MocModel, DrawableInfo
from .meshwarp import premultiply, straight_u8, rasterize_triangles, render_drawable


@dataclass
class BakeOptions:
    bleed_radius: int = 2
    backup_original: bool = True
    only_modified: bool = True
    output_dir: Optional[str] = None
    expression: Optional[str] = None


def bake_psd_to_atlases(
    psd_path: str,
    model: MocModel,
    options: Optional[BakeOptions] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    if options is None:
        options = BakeOptions()

    if options.expression:
        model.apply_expression(options.expression)
    elif not model.current_expression:
        meta_path = f"{psd_path}.meta.json"
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                exp = meta.get("expression")
                if exp:
                    model.apply_expression(exp)
            except Exception:
                pass

    start = time.time()
    if not os.path.isfile(psd_path):
        raise FileNotFoundError(f"PSD not found: {psd_path}")
    if progress_callback:
        progress_callback(0, 100, "Reading PSD")

    psd = PSDImage.open(psd_path)
    canvas = model.canvas
    scale = psd.width / canvas.width
    size = (psd.width, psd.height)

    all_layers = _collect_leaf_layers(psd)
    total = len(all_layers)
    pms = [premultiply(t) for t in model.textures]
    atlases = [t.copy() for t in model.textures]
    mod_masks = [np.zeros(t.shape[:2], bool) for t in model.textures]
    baked = unchanged = skipped = 0

    for idx, layer in enumerate(all_layers):
        if progress_callback:
            progress_callback(idx + 1, total, f"Checking {layer.name}")
        d = _find_matching_drawable(layer.name, model.drawable_id_map)
        if d is None or d.texture_index < 0 or d.texture_index >= len(atlases):
            skipped += 1
            continue
        frame = render_drawable(d, canvas, pms, scale, size)
        if frame is None:
            skipped += 1
            continue

        pil = None
        has_pixels = getattr(layer, "has_pixels", lambda: True)()
        if has_pixels and getattr(layer, "width", 1) > 0 and getattr(layer, "height", 1) > 0:
            try:
                pil = layer.topil()
            except Exception:
                pil = None

        ed = np.zeros_like(frame.img)
        if pil is not None:
            rgba = np.array(pil.convert("RGBA"))
            bgra = rgba[..., [2, 1, 0, 3]]
            lx, ly = layer.left, layer.top
            lh, lw = bgra.shape[:2]
            ix0, iy0 = max(lx, frame.x0), max(ly, frame.y0)
            ix1, iy1 = min(lx + lw, frame.x1), min(ly + lh, frame.y1)
            if ix1 > ix0 and iy1 > iy0:
                ed[iy0 - frame.y0 : iy1 - frame.y0, ix0 - frame.x0 : ix1 - frame.x0] = (
                    bgra[iy0 - ly : iy1 - ly, ix0 - lx : ix1 - lx]
                )

        is_cleared = (pil is None) or not (ed[..., 3] > 0).any()
        ed_pm = premultiply(ed)
        if options.only_modified:
            diff = np.abs(premultiply(frame.img) - ed_pm).max(axis=2) > (3 / 255.0)
            if not diff.any():
                unchanged += 1
                continue
            changed = cv2.dilate(diff.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
        else:
            changed = np.ones(ed.shape[:2], np.float32)

        atlas = atlases[d.texture_index]
        th, tw = atlas.shape[:2]
        uv = d.get_atlas_pixel_coords(tw, th).astype(np.float64)
        pos = d.get_canvas_positions(canvas).astype(np.float64) * scale - [frame.x0, frame.y0]

        ax0 = max(0, int(np.floor(uv[:, 0].min())) - 1)
        ay0 = max(0, int(np.floor(uv[:, 1].min())) - 1)
        ax1 = min(tw, int(np.ceil(uv[:, 0].max())) + 1)
        ay1 = min(th, int(np.ceil(uv[:, 1].max())) + 1)
        tri = d.indices.reshape(-1, 3)
        mx, my, cover = rasterize_triangles(uv[tri] - [ax0, ay0], pos[tri] - 0.5, ax1 - ax0, ay1 - ay0)

        if is_cleared:
            sel = cover
            samp = np.zeros((ay1 - ay0, ax1 - ax0, 4), dtype=np.float32)
        else:
            samp = cv2.remap(ed_pm, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            chg = cv2.remap(changed, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.01
            sel = cover & chg

        region = atlas[ay0:ay1, ax0:ax1]
        region[sel] = straight_u8(samp)[sel]
        mod_masks[d.texture_index][ay0:ay1, ax0:ax1] |= sel
        baked += 1

    if options.bleed_radius > 0:
        if progress_callback:
            progress_callback(total, total, "Padding edges")
        for a, m in zip(atlases, mod_masks):
            bleed_mask = m & (a[..., 3] > 0)
            if bleed_mask.any():
                _apply_edge_bleed(a, bleed_mask, options.bleed_radius)

    out_dir = options.output_dir or (
        os.path.dirname(model.texture_paths[0]) if model.texture_paths else model.model_dir
    )
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for i, orig in enumerate(model.texture_paths):
        dest = os.path.join(out_dir, os.path.basename(orig))
        if options.backup_original and os.path.abspath(dest) == os.path.abspath(orig):
            bak = f"{orig}.bak"
            if not os.path.isfile(bak):
                shutil.copy2(orig, bak)
        cv2.imwrite(dest, atlases[i])
        written.append(dest)

    return {
        "baked_layers": baked,
        "unchanged_layers": unchanged,
        "skipped_layers": skipped,
        "total_layers": total,
        "written_textures": written,
        "elapsed_seconds": time.time() - start,
    }


def _collect_leaf_layers(group) -> list:
    leaves = []
    for item in group:
        if item.is_group():
            leaves.extend(_collect_leaf_layers(item))
        else:
            leaves.append(item)
    return leaves


def _find_matching_drawable(layer_name: str, id_map: Dict[str, DrawableInfo]) -> Optional[DrawableInfo]:
    if layer_name in id_map:
        return id_map[layer_name]
    m = re.search(r"\[.*?\]\s*([A-Za-z0-9_]+)", layer_name)
    if m and m.group(1) in id_map:
        return id_map[m.group(1)]
    return None


def _apply_edge_bleed(atlas: np.ndarray, modified: np.ndarray, radius: int = 2):
    k = 2 * radius + 1
    near = cv2.dilate(modified.astype(np.uint8), np.ones((k + 2, k + 2), np.uint8)) > 0
    rgb = atlas[..., :3].astype(np.float32)
    known = (atlas[..., 3] > 0).astype(np.float32)
    for _ in range(radius):
        s = cv2.boxFilter(rgb * known[..., None], -1, (3, 3), normalize=False)
        c = cv2.boxFilter(known, -1, (3, 3), normalize=False)
        fill = (known == 0) & (c > 0) & near
        if not fill.any():
            break
        rgb[fill] = s[fill] / c[fill][:, None]
        known[fill] = 1.0
    out = atlas[..., :3]
    changed = (atlas[..., 3] == 0) & near & (known > 0)
    out[changed] = np.clip(rgb[changed] + 0.5, 0, 255).astype(np.uint8)
