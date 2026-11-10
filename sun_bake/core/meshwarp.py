from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .model import MocModel, DrawableInfo, CanvasInfo


@dataclass
class RenderedLayer:
    x0: int
    y0: int
    img: np.ndarray

    @property
    def x1(self) -> int:
        return self.x0 + self.img.shape[1]

    @property
    def y1(self) -> int:
        return self.y0 + self.img.shape[0]


def premultiply(tex_bgra: np.ndarray) -> np.ndarray:
    f = tex_bgra.astype(np.float32) * (1.0 / 255.0)
    f[..., :3] *= f[..., 3:4]
    return f


def straight_u8(pm: np.ndarray) -> np.ndarray:
    a = pm[..., 3:4]
    rgb = np.where(a > 1e-6, pm[..., :3] / np.maximum(a, 1e-6), 0.0)
    out = np.concatenate([rgb, a], axis=-1)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def rasterize_triangles(
    tri_pos: np.ndarray, tri_val: np.ndarray, w: int, h: int, eps: float = 1e-3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mx = np.zeros((h, w), np.float32)
    my = np.zeros((h, w), np.float32)
    cover = np.zeros((h, w), bool)
    for p, v in zip(tri_pos, tri_val):
        x0 = max(int(np.floor(p[:, 0].min() - 0.5)), 0)
        y0 = max(int(np.floor(p[:, 1].min() - 0.5)), 0)
        x1 = min(int(np.ceil(p[:, 0].max() + 0.5)) + 1, w)
        y1 = min(int(np.ceil(p[:, 1].max() + 0.5)) + 1, h)
        if x1 <= x0 or y1 <= y0:
            continue
        d1 = p[1] - p[0]
        d2 = p[2] - p[0]
        det = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(det) < 1e-9:
            continue
        xs = (np.arange(x0, x1) + 0.5 - p[0, 0])[None, :]
        ys = (np.arange(y0, y1) + 0.5 - p[0, 1])[:, None]
        l1 = (xs * d2[1] - ys * d2[0]) / det
        l2 = (d1[0] * ys - d1[1] * xs) / det
        l0 = 1.0 - l1 - l2
        inside = (l0 >= -eps) & (l1 >= -eps) & (l2 >= -eps)
        if not inside.any():
            continue
        e1 = v[1] - v[0]
        e2 = v[2] - v[0]
        vx = v[0, 0] + l1 * e1[0] + l2 * e2[0]
        vy = v[0, 1] + l1 * e1[1] + l2 * e2[1]
        sl = (slice(y0, y1), slice(x0, x1))
        mx[sl][inside] = vx[inside]
        my[sl][inside] = vy[inside]
        cover[sl] |= inside
    return mx, my, cover


def uv_area(d: DrawableInfo, tw: int, th: int) -> float:
    if len(d.indices) < 3:
        return 0.0
    uv = d.get_atlas_pixel_coords(tw, th).astype(np.float64)
    t = d.indices.reshape(-1, 3)
    a = uv[t[:, 1]] - uv[t[:, 0]]
    b = uv[t[:, 2]] - uv[t[:, 0]]
    return float(np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum() * 0.5)


def auto_scale(model: MocModel) -> float:
    ratios = []
    for d in model.drawables:
        if d.texture_index < 0 or d.texture_index >= len(model.textures):
            continue
        th, tw = model.textures[d.texture_index].shape[:2]
        ua = uv_area(d, tw, th)
        if ua < 64:
            continue
        pos = d.get_canvas_positions(model.canvas).astype(np.float64)
        t = d.indices.reshape(-1, 3)
        a = pos[t[:, 1]] - pos[t[:, 0]]
        b = pos[t[:, 2]] - pos[t[:, 0]]
        pa = float(np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum() * 0.5)
        ratios.append(np.sqrt(pa / ua))
    if not ratios:
        return 1.0
    return round(1.0 / float(np.median(ratios)), 4)


def canvas_size_for(model: MocModel, scale: float) -> Tuple[int, int]:
    return int(round(model.canvas.width * scale)), int(round(model.canvas.height * scale))


def render_drawable(
    d: DrawableInfo,
    canvas: CanvasInfo,
    tex_pm_list: List[np.ndarray],
    scale: float,
    canvas_size: Tuple[int, int],
) -> Optional[RenderedLayer]:
    if d.texture_index < 0 or d.texture_index >= len(tex_pm_list) or len(d.indices) < 3:
        return None
    tex_pm = tex_pm_list[d.texture_index]
    th, tw = tex_pm.shape[:2]
    if uv_area(d, tw, th) < 1.0:
        return None

    cw, ch = canvas_size
    pos = d.get_canvas_positions(canvas).astype(np.float64) * scale
    uv = d.get_atlas_pixel_coords(tw, th).astype(np.float64)

    x0 = max(0, int(np.floor(pos[:, 0].min())) - 1)
    y0 = max(0, int(np.floor(pos[:, 1].min())) - 1)
    x1 = min(cw, int(np.ceil(pos[:, 0].max())) + 1)
    y1 = min(ch, int(np.ceil(pos[:, 1].max())) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    w, h = x1 - x0, y1 - y0

    tri = d.indices.reshape(-1, 3)
    mx, my, cover = rasterize_triangles(pos[tri] - [x0, y0], uv[tri] - 0.5, w, h)
    if not cover.any():
        return None
    pm = cv2.remap(tex_pm, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    pm[~cover] = 0
    img = straight_u8(pm)
    if img[..., 3].max() == 0:
        return None
    return RenderedLayer(x0, y0, img)


def build_layers(
    model: MocModel,
    scale: float,
    textures: Optional[List[np.ndarray]] = None,
    progress=None,
) -> Dict[int, RenderedLayer]:
    textures = textures if textures is not None else model.textures
    pms = [premultiply(t) for t in textures]
    size = canvas_size_for(model, scale)
    out: Dict[int, RenderedLayer] = {}
    n = len(model.drawables)
    for i, d in enumerate(model.drawables):
        if progress:
            progress(i + 1, n, f"Rasterizing {d.id} ({i + 1}/{n})")
        r = render_drawable(d, model.canvas, pms, scale, size)
        if r is not None:
            out[d.index] = r
    return out


def compute_clip_mask(d: DrawableInfo, layers: Dict[int, RenderedLayer]) -> Optional[np.ndarray]:
    L = layers.get(d.index)
    if L is None or not d.mask_indices:
        return None
    h, w = L.img.shape[:2]
    acc = np.zeros((h, w), np.float32)
    for mi in d.mask_indices:
        M = layers.get(mi)
        if M is None:
            continue
        ix0, iy0 = max(L.x0, M.x0), max(L.y0, M.y0)
        ix1, iy1 = min(L.x1, M.x1), min(L.y1, M.y1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        acc[iy0 - L.y0 : iy1 - L.y0, ix0 - L.x0 : ix1 - L.x0] += (
            M.img[iy0 - M.y0 : iy1 - M.y0, ix0 - M.x0 : ix1 - M.x0, 3].astype(np.float32) * (1.0 / 255.0)
        )
    acc = np.minimum(acc, 1.0)
    if d.is_inverted_mask:
        acc = 1.0 - acc
    return (acc * 255.0 + 0.5).astype(np.uint8)
