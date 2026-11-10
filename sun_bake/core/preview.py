from typing import Optional, List
import numpy as np

from .model import MocModel
from .meshwarp import build_layers, compute_clip_mask, premultiply, canvas_size_for


def render_model_preview(
    model: MocModel,
    scale: float = 0.25,
    custom_textures: Optional[List[np.ndarray]] = None,
    expression: Optional[str] = None,
) -> np.ndarray:
    if expression:
        model.apply_expression(expression)

    layers = build_layers(model, scale, custom_textures)
    pw, ph = canvas_size_for(model, scale)
    out = np.zeros((ph, pw, 4), np.float32)

    for d in sorted(model.drawables, key=lambda x: x.render_order):
        if not d.is_visible or d.opacity <= 0.0:
            continue
        L = layers.get(d.index)
        if L is None:
            continue
        pm = premultiply(L.img)
        mask = compute_clip_mask(d, layers)
        if mask is not None:
            pm *= (mask.astype(np.float32) * (1.0 / 255.0))[..., None]
        pm *= d.opacity

        x0, y0, x1, y1 = L.x0, L.y0, L.x1, L.y1
        dst = out[y0:y1, x0:x1]
        sa = pm[..., 3:4]
        if d.is_additive:
            dst[..., :3] += pm[..., :3]
            dst[..., 3:4] = sa + dst[..., 3:4] * (1 - sa)
        elif d.is_multiply:
            da = dst[..., 3:4]
            dst[..., :3] = pm[..., :3] * dst[..., :3] + pm[..., :3] * (1 - da) + dst[..., :3] * (1 - sa)
            dst[..., 3:4] = sa + da * (1 - sa)
        else:
            dst[...] = pm + dst * (1 - sa)

    a = out[..., 3:4]
    rgb = np.where(a > 1e-6, out[..., :3] / np.maximum(a, 1e-6), 0)
    res = np.concatenate([rgb, a], axis=-1)
    return np.clip(res * 255.0 + 0.5, 0, 255).astype(np.uint8)
