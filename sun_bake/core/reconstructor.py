import os
import json
import time
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, List
import numpy as np
import cv2

import packbits
import pytoshop.codecs
pytoshop.codecs.packbits = packbits

from pytoshop import core, layers, enums, tagged_block
from .model import MocModel, DrawableInfo
from .meshwarp import (
    build_layers,
    compute_clip_mask,
    auto_scale,
    canvas_size_for,
)


@dataclass
class ExportOptions:
    group_by_part: bool = True
    include_invisible: bool = True
    write_metadata: bool = True
    write_masks: bool = True
    scale: Optional[float] = None
    compression: enums.Compression = enums.Compression.rle
    expression: Optional[str] = None


class _Group:
    def __init__(self, name: str, items: Optional[list] = None):
        self.name = name
        self.items: list = items if items is not None else []


def _build_tree(model: MocModel, drawables: List[DrawableInfo]) -> List:
    groups: List = []
    curr_part_name: Optional[str] = None
    curr_items: List[DrawableInfo] = []

    for d in drawables:
        p_name = model.parts[d.parent_part_index].id if d.parent_part_index >= 0 else ""
        if p_name != curr_part_name:
            if curr_items:
                if curr_part_name:
                    groups.append(_Group(curr_part_name, curr_items))
                else:
                    groups.extend(curr_items)
            curr_part_name = p_name
            curr_items = [d]
        else:
            curr_items.append(d)

    if curr_items:
        if curr_part_name:
            groups.append(_Group(curr_part_name, curr_items))
        else:
            groups.extend(curr_items)

    return groups


def export_model_to_psd(
    model: MocModel,
    output_psd_path: str,
    options: Optional[ExportOptions] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    if options is None:
        options = ExportOptions()

    if options.expression:
        if progress_callback:
            progress_callback(0, 100, f"Applying expression {options.expression}")
        model.apply_expression(options.expression)

    start = time.time()
    scale = options.scale if options.scale else auto_scale(model)
    can_w, can_h = canvas_size_for(model, scale)

    rendered = build_layers(model, scale, progress=progress_callback)

    psd = core.PsdFile(num_channels=3, height=can_h, width=can_w, depth=8, color_mode=enums.ColorMode.rgb)
    records = psd.layer_and_mask_info.layer_info.layer_records
    meta_drawables: Dict[str, Any] = {}
    exported = 0
    total = len(rendered)

    def make_record(d: DrawableInfo) -> layers.LayerRecord:
        L = rendered[d.index]
        b, g, r, a = cv2.split(L.img)
        comp = options.compression
        ch = {
            0: layers.ChannelImageData(image=r, compression=comp),
            1: layers.ChannelImageData(image=g, compression=comp),
            2: layers.ChannelImageData(image=b, compression=comp),
            -1: layers.ChannelImageData(image=a, compression=comp),
        }
        mask = compute_clip_mask(d, rendered) if options.write_masks else None
        if mask is not None:
            ch[-2] = layers.ChannelImageData(image=mask, compression=comp)

        blend = enums.BlendMode.normal
        if d.is_additive:
            blend = enums.BlendMode.linear_dodge
        elif d.is_multiply:
            blend = enums.BlendMode.multiply
        hidden = d.opacity <= 0.0 or not d.is_visible
        rec = layers.LayerRecord(
            name=d.id,
            top=L.y0, left=L.x0, bottom=L.y1, right=L.x1,
            blend_mode=blend,
            opacity=255 if hidden else int(round(np.clip(d.opacity, 0, 1) * 255)),
            visible=not hidden,
            channels=ch,
        )
        if mask is not None:
            rec.mask = layers.LayerMask(
                top=L.y0, left=L.x0, bottom=L.y1, right=L.x1,
                default_color=bool(d.is_inverted_mask),
            )
        part = model.parts[d.parent_part_index].id if d.parent_part_index >= 0 else ""
        meta_drawables[d.id] = {
            "index": d.index,
            "texture_index": d.texture_index,
            "render_order": d.render_order,
            "bbox": [L.x0, L.y0, L.x1, L.y1],
            "parent_part": part,
            "masks": [model.drawables[m].id for m in d.mask_indices],
            "mask_inverted": d.is_inverted_mask,
            "blend": str(blend),
            "default_opacity": d.opacity,
            "default_visible": d.is_visible,
        }
        return rec

    def emit(items: list):
        nonlocal exported
        for it in items:
            if isinstance(it, _Group):
                records.append(layers.LayerRecord(
                    name="</Layer set>", top=0, left=0, bottom=0, right=0, channels={},
                    blocks=[tagged_block.SectionDividerSetting(type=enums.SectionDividerSetting.bounding)],
                ))
                emit(it.items)
                records.append(layers.LayerRecord(
                    name=it.name, top=0, left=0, bottom=0, right=0, channels={},
                    blocks=[tagged_block.SectionDividerSetting(type=enums.SectionDividerSetting.closed)],
                ))
            else:
                records.append(make_record(it))
                exported += 1
                if progress_callback:
                    progress_callback(exported, total, f"Packing {it.id}")

    drawables = sorted((d for d in model.drawables if d.index in rendered), key=lambda d: d.render_order)
    if options.group_by_part:
        emit(_build_tree(model, drawables))
    else:
        emit(drawables)

    if progress_callback:
        progress_callback(total, total, "Writing PSD")
    os.makedirs(os.path.dirname(os.path.abspath(output_psd_path)), exist_ok=True)
    with open(output_psd_path, "wb") as f:
        psd.write(f)

    if options.write_metadata:
        base = os.path.dirname(os.path.abspath(output_psd_path))
        with open(f"{output_psd_path}.meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "model_moc": os.path.relpath(model.moc_path, base),
                "texture_paths": [os.path.relpath(t, base) for t in model.texture_paths],
                "expression": model.current_expression,
                "canvas": {
                    "width": model.canvas.width, "height": model.canvas.height,
                    "origin_x": model.canvas.origin_x, "origin_y": model.canvas.origin_y,
                    "ppu": model.canvas.ppu,
                },
                "psd_scale": scale,
                "psd_size": [can_w, can_h],
                "exported_layers": exported,
                "drawables": meta_drawables,
            }, f, indent=2)

    return {
        "output_psd_path": output_psd_path,
        "exported_layers": exported,
        "skipped_layers": len(model.drawables) - exported,
        "total_drawables": len(model.drawables),
        "file_size_mb": os.path.getsize(output_psd_path) / (1024 * 1024),
        "elapsed_seconds": time.time() - start,
        "scale": scale,
        "expression": model.current_expression,
    }
