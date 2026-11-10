import os
import json
import ctypes
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import cv2

from .cubism_core import (
    CubismCore,
    CSM_ALIGN_OF_MOC,
    CSM_ALIGN_OF_MODEL,
    csmVector2,
)


@dataclass
class CanvasInfo:
    width: float
    height: float
    origin_x: float
    origin_y: float
    ppu: float

    def model_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        return self.origin_x + x * self.ppu, self.origin_y - y * self.ppu

    def canvas_to_model(self, cx: float, cy: float) -> Tuple[float, float]:
        return (cx - self.origin_x) / self.ppu, (self.origin_y - cy) / self.ppu


@dataclass
class PartInfo:
    index: int
    id: str
    parent_index: int
    opacity: float


@dataclass
class DrawableInfo:
    index: int
    id: str
    texture_index: int
    render_order: int
    draw_order: int
    opacity: float
    is_visible: bool
    is_additive: bool
    is_multiply: bool
    is_double_sided: bool
    is_inverted_mask: bool
    parent_part_index: int
    mask_indices: List[int]
    vertices: np.ndarray
    uvs: np.ndarray
    indices: np.ndarray

    def get_canvas_positions(self, canvas: CanvasInfo) -> np.ndarray:
        xs = canvas.origin_x + self.vertices[:, 0] * canvas.ppu
        ys = canvas.origin_y - self.vertices[:, 1] * canvas.ppu
        return np.column_stack((xs, ys)).astype(np.float32)

    def get_atlas_pixel_coords(self, atlas_w: int, atlas_h: int) -> np.ndarray:
        tx = self.uvs[:, 0] * atlas_w
        ty = (1.0 - self.uvs[:, 1]) * atlas_h
        return np.column_stack((tx, ty)).astype(np.float32)

    def get_canvas_bbox(self, canvas: CanvasInfo) -> Tuple[int, int, int, int]:
        pos = self.get_canvas_positions(canvas)
        return (
            int(np.floor(pos[:, 0].min())),
            int(np.floor(pos[:, 1].min())),
            int(np.ceil(pos[:, 0].max())),
            int(np.ceil(pos[:, 1].max())),
        )


@dataclass
class ParameterInfo:
    index: int
    id: str
    value: float
    default_value: float
    min_value: float
    max_value: float


def _blend_param(cur: float, val: float, blend: Any) -> float:
    if isinstance(blend, str):
        b = blend.strip().lower()
        if b in ("overwrite", "override", "set"):
            return val
        if b == "multiply":
            return cur * val
        return cur + val
    if blend == 2:
        return val
    if blend == 0:
        return cur * val
    return cur + val


class MocModel:
    def __init__(
        self,
        moc_path: str,
        texture_paths: Optional[List[str]] = None,
        dll_path: Optional[str] = None,
        expressions: Optional[Dict[str, str]] = None,
        expression: Optional[str] = None,
    ):
        self.moc_path = os.path.abspath(moc_path)
        self.model_dir = os.path.dirname(self.moc_path)
        self.core = CubismCore.get_instance(dll_path)

        self._moc_buffer = None
        self._model_buffer = None
        self._moc_ptr = None
        self._model_ptr = None

        self.canvas: Optional[CanvasInfo] = None
        self.parameters: List[ParameterInfo] = []
        self.parameter_id_map: Dict[str, ParameterInfo] = {}
        self.parts: List[PartInfo] = []
        self.drawables: List[DrawableInfo] = []
        self.drawable_id_map: Dict[str, DrawableInfo] = {}

        self.expressions: Dict[str, str] = dict(expressions) if expressions else {}
        self.current_expression: Optional[str] = None

        self.texture_paths: List[str] = (
            [os.path.abspath(p) for p in texture_paths] if texture_paths else []
        )
        self.textures: List[np.ndarray] = []

        self._load_and_initialize()

        if expression:
            self.apply_expression(expression)

    @classmethod
    def from_model3_json(
        cls,
        model3_json_path: str,
        dll_path: Optional[str] = None,
        expression: Optional[str] = None,
    ) -> "MocModel":
        model3_json_path = os.path.abspath(model3_json_path)
        base_dir = os.path.dirname(model3_json_path)

        with open(model3_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        file_refs = data.get("FileReferences", {})
        moc_rel = file_refs.get("Moc")
        if not moc_rel:
            raise ValueError(f"Missing 'Moc' entry in {model3_json_path}")

        moc_path = os.path.join(base_dir, moc_rel)
        tex_rels = file_refs.get("Textures", [])
        texture_paths = [os.path.join(base_dir, t) for t in tex_rels]

        expressions: Dict[str, str] = {}
        for exp_entry in file_refs.get("Expressions", []):
            name = exp_entry.get("Name", "")
            f_rel = exp_entry.get("File", "")
            if f_rel:
                exp_full = os.path.normpath(os.path.join(base_dir, f_rel))
                if os.path.isfile(exp_full):
                    if name:
                        expressions[name] = exp_full
                    expressions[os.path.basename(exp_full)] = exp_full

        return cls(
            moc_path=moc_path,
            texture_paths=texture_paths,
            dll_path=dll_path,
            expressions=expressions,
            expression=expression,
        )

    def _load_and_initialize(self):
        c = self.core._dll

        with open(self.moc_path, "rb") as f:
            moc_bytes = f.read()

        moc_len = len(moc_bytes)
        self._moc_buffer = (ctypes.c_char * (moc_len + CSM_ALIGN_OF_MOC))()
        raw_addr = ctypes.addressof(self._moc_buffer)
        aligned_moc_addr = (raw_addr + CSM_ALIGN_OF_MOC - 1) & ~(CSM_ALIGN_OF_MOC - 1)
        ctypes.memmove(aligned_moc_addr, moc_bytes, moc_len)

        self._moc_ptr = c.csmReviveMocInPlace(aligned_moc_addr, moc_len)
        if not self._moc_ptr:
            raise RuntimeError(f"csmReviveMocInPlace failed for {self.moc_path}")

        model_size = c.csmGetSizeofModel(self._moc_ptr)
        self._model_buffer = (ctypes.c_char * (model_size + CSM_ALIGN_OF_MODEL))()
        raw_m_addr = ctypes.addressof(self._model_buffer)
        aligned_model_addr = (raw_m_addr + CSM_ALIGN_OF_MODEL - 1) & ~(CSM_ALIGN_OF_MODEL - 1)

        self._model_ptr = c.csmInitializeModelInPlace(self._moc_ptr, aligned_model_addr, model_size)
        if not self._model_ptr:
            raise RuntimeError(f"csmInitializeModelInPlace failed for {self.moc_path}")

        c.csmUpdateModel(self._model_ptr)

        size_in_pixels = csmVector2()
        origin_in_pixels = csmVector2()
        ppu = ctypes.c_float()
        c.csmReadCanvasInfo(
            self._model_ptr,
            ctypes.byref(size_in_pixels),
            ctypes.byref(origin_in_pixels),
            ctypes.byref(ppu),
        )
        self.canvas = CanvasInfo(
            width=size_in_pixels.X,
            height=size_in_pixels.Y,
            origin_x=origin_in_pixels.X,
            origin_y=origin_in_pixels.Y,
            ppu=ppu.value,
        )

        param_count = c.csmGetParameterCount(self._model_ptr)
        param_ids_ptr = c.csmGetParameterIds(self._model_ptr)
        param_vals_ptr = c.csmGetParameterValues(self._model_ptr)
        param_defs_ptr = c.csmGetParameterDefaultValues(self._model_ptr)
        param_mins_ptr = c.csmGetParameterMinimumValues(self._model_ptr)
        param_maxs_ptr = c.csmGetParameterMaximumValues(self._model_ptr)

        self.parameters = []
        self.parameter_id_map = {}
        for p in range(param_count):
            p_id = param_ids_ptr[p].decode("utf-8") if param_ids_ptr[p] else f"Param_{p}"
            p_info = ParameterInfo(
                index=p,
                id=p_id,
                value=float(param_vals_ptr[p]),
                default_value=float(param_defs_ptr[p]),
                min_value=float(param_mins_ptr[p]),
                max_value=float(param_maxs_ptr[p]),
            )
            self.parameters.append(p_info)
            self.parameter_id_map[p_id] = p_info

        part_count = c.csmGetPartCount(self._model_ptr)
        part_ids_ptr = c.csmGetPartIds(self._model_ptr)
        part_opacities = c.csmGetPartOpacities(self._model_ptr)
        part_parents = c.csmGetPartParentPartIndices(self._model_ptr)

        self.parts = []
        for p in range(part_count):
            p_id = part_ids_ptr[p].decode("utf-8") if part_ids_ptr[p] else f"Part_{p}"
            self.parts.append(
                PartInfo(
                    index=p,
                    id=p_id,
                    parent_index=part_parents[p],
                    opacity=part_opacities[p],
                )
            )

        d_count = c.csmGetDrawableCount(self._model_ptr)
        ids_ptr = c.csmGetDrawableIds(self._model_ptr)
        c_flags = c.csmGetDrawableConstantFlags(self._model_ptr)
        d_flags = c.csmGetDrawableDynamicFlags(self._model_ptr)
        tex_indices = c.csmGetDrawableTextureIndices(self._model_ptr)
        draw_orders = c.csmGetDrawableDrawOrders(self._model_ptr)
        render_orders = c.csmGetRenderOrders(self._model_ptr)
        opacities = c.csmGetDrawableOpacities(self._model_ptr)
        mask_counts = c.csmGetDrawableMaskCounts(self._model_ptr)
        masks_ptr = c.csmGetDrawableMasks(self._model_ptr)
        v_counts = c.csmGetDrawableVertexCounts(self._model_ptr)
        pos_ptr = c.csmGetDrawableVertexPositions(self._model_ptr)
        uv_ptr = c.csmGetDrawableVertexUvs(self._model_ptr)
        idx_counts = c.csmGetDrawableIndexCounts(self._model_ptr)
        indices_ptr = c.csmGetDrawableIndices(self._model_ptr)
        parents_ptr = c.csmGetDrawableParentPartIndices(self._model_ptr)

        self.drawables = []
        self.drawable_id_map = {}

        for i in range(d_count):
            d_id = ids_ptr[i].decode("utf-8") if ids_ptr[i] else f"ArtMesh_{i}"
            nv = v_counts[i]
            nidx = idx_counts[i]

            v_arr = np.zeros((nv, 2), dtype=np.float32)
            if nv > 0 and pos_ptr[i]:
                flt_ptr = ctypes.cast(pos_ptr[i], ctypes.POINTER(ctypes.c_float))
                v_arr = np.copy(np.ctypeslib.as_array(flt_ptr, shape=(nv, 2)))

            uv_arr = np.zeros((nv, 2), dtype=np.float32)
            if nv > 0 and uv_ptr[i]:
                uv_flt = ctypes.cast(uv_ptr[i], ctypes.POINTER(ctypes.c_float))
                uv_arr = np.copy(np.ctypeslib.as_array(uv_flt, shape=(nv, 2)))

            i_arr = np.zeros(nidx, dtype=np.int32)
            if nidx > 0 and indices_ptr[i]:
                i_arr = np.ctypeslib.as_array(indices_ptr[i], shape=(nidx,)).astype(np.int32, copy=True)

            mc = mask_counts[i]
            mask_list = [masks_ptr[i][m] for m in range(mc)] if mc > 0 and masks_ptr[i] else []

            cf = c_flags[i]
            df = d_flags[i]

            item = DrawableInfo(
                index=i,
                id=d_id,
                texture_index=tex_indices[i],
                render_order=render_orders[i],
                draw_order=draw_orders[i],
                opacity=opacities[i],
                is_visible=bool(df & 1),
                is_additive=bool(cf & 1),
                is_multiply=bool(cf & 2),
                is_double_sided=bool(cf & 4),
                is_inverted_mask=bool(cf & 8),
                parent_part_index=parents_ptr[i],
                mask_indices=mask_list,
                vertices=v_arr,
                uvs=uv_arr,
                indices=i_arr,
            )
            self.drawables.append(item)
            self.drawable_id_map[d_id] = item

        self._discover_expressions()
        if not self.texture_paths:
            self._discover_textures()
        self.load_textures()

    def _sync_model_state(self):
        c = self.core._dll
        pos_ptr = c.csmGetDrawableVertexPositions(self._model_ptr)
        v_counts = c.csmGetDrawableVertexCounts(self._model_ptr)
        opacities = c.csmGetDrawableOpacities(self._model_ptr)
        render_orders = c.csmGetRenderOrders(self._model_ptr)
        draw_orders = c.csmGetDrawableDrawOrders(self._model_ptr)
        d_flags = c.csmGetDrawableDynamicFlags(self._model_ptr)

        for d in self.drawables:
            nv = v_counts[d.index]
            if nv > 0 and pos_ptr[d.index]:
                flt_ptr = ctypes.cast(pos_ptr[d.index], ctypes.POINTER(ctypes.c_float))
                d.vertices = np.copy(np.ctypeslib.as_array(flt_ptr, shape=(nv, 2)))
            d.opacity = float(opacities[d.index])
            d.render_order = int(render_orders[d.index])
            d.draw_order = int(draw_orders[d.index])
            d.is_visible = bool(d_flags[d.index] & 1)

        part_opacities = c.csmGetPartOpacities(self._model_ptr)
        for part in self.parts:
            part.opacity = float(part_opacities[part.index])

        param_vals = c.csmGetParameterValues(self._model_ptr)
        for p in self.parameters:
            p.value = float(param_vals[p.index])

    def _discover_expressions(self):
        candidates = []
        exp_dir = os.path.join(self.model_dir, "expressions")
        if os.path.isdir(exp_dir):
            for f in sorted(os.listdir(exp_dir)):
                if f.lower().endswith((".exp3.json", ".json")):
                    candidates.append(os.path.join(exp_dir, f))
        for f in sorted(os.listdir(self.model_dir)):
            if f.lower().endswith(".exp3.json"):
                candidates.append(os.path.join(self.model_dir, f))

        for full_p in candidates:
            base = os.path.basename(full_p)
            name = base[:-10] if base.lower().endswith(".exp3.json") else os.path.splitext(base)[0]
            self.expressions.setdefault(name, full_p)
            self.expressions.setdefault(base, full_p)

    def _discover_textures(self):
        tex_dir = os.path.join(self.model_dir, "textures")
        candidates = []
        if os.path.isdir(tex_dir):
            for f in sorted(os.listdir(tex_dir)):
                if f.lower().endswith(".png"):
                    candidates.append(os.path.join(tex_dir, f))

        if not candidates:
            for f in sorted(os.listdir(self.model_dir)):
                if f.lower().endswith(".png"):
                    candidates.append(os.path.join(self.model_dir, f))

        self.texture_paths = candidates

    def load_textures(self):
        self.textures = []
        for p in self.texture_paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Texture not found: {p}")
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise IOError(f"Could not read image: {p}")
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            self.textures.append(img)

    def get_parameter_value(self, param_id_or_index: Any) -> float:
        if isinstance(param_id_or_index, int):
            return self.parameters[param_id_or_index].value
        return self.parameter_id_map[param_id_or_index].value

    def set_parameter_value(
        self,
        param_id_or_index: Any,
        value: float,
        blend: Any = "Overwrite",
        auto_update: bool = True,
    ) -> float:
        c = self.core._dll
        p_info = (
            self.parameters[param_id_or_index]
            if isinstance(param_id_or_index, int)
            else self.parameter_id_map[param_id_or_index]
        )
        vals_ptr = c.csmGetParameterValues(self._model_ptr)
        new_val = _blend_param(vals_ptr[p_info.index], value, blend)
        new_val = max(p_info.min_value, min(p_info.max_value, new_val))
        vals_ptr[p_info.index] = new_val
        p_info.value = new_val

        if auto_update:
            c.csmUpdateModel(self._model_ptr)
            self._sync_model_state()

        return new_val

    def reset_parameters(self):
        c = self.core._dll
        vals_ptr = c.csmGetParameterValues(self._model_ptr)
        for p in self.parameters:
            vals_ptr[p.index] = p.default_value
            p.value = p.default_value

        c.csmUpdateModel(self._model_ptr)
        self._sync_model_state()
        self.current_expression = None

    def resolve_expression_path(self, expr_arg: str) -> str:
        if not expr_arg:
            raise ValueError("Empty expression argument")
        if os.path.isfile(expr_arg):
            return os.path.abspath(expr_arg)
        rel_model = os.path.join(self.model_dir, expr_arg)
        if os.path.isfile(rel_model):
            return os.path.abspath(rel_model)
        if expr_arg in self.expressions:
            return self.expressions[expr_arg]
        with_ext = f"{expr_arg}.exp3.json"
        if with_ext in self.expressions:
            return self.expressions[with_ext]
        in_exp_dir = os.path.join(self.model_dir, "expressions", expr_arg)
        if os.path.isfile(in_exp_dir):
            return os.path.abspath(in_exp_dir)
        in_exp_dir_ext = os.path.join(self.model_dir, "expressions", f"{expr_arg}.exp3.json")
        if os.path.isfile(in_exp_dir_ext):
            return os.path.abspath(in_exp_dir_ext)

        avail = ", ".join(sorted(self.expressions.keys())) or "None"
        raise FileNotFoundError(f"Expression '{expr_arg}' not found (available: {avail})")

    def apply_expression(self, expression: Any, reset_first: bool = True) -> Dict[str, Any]:
        if isinstance(expression, dict):
            data = expression
            expr_name = data.get("Name", "custom_expression")
            expr_source = "dict"
        else:
            resolved_path = self.resolve_expression_path(str(expression))
            with open(resolved_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            expr_name = os.path.basename(resolved_path)
            expr_source = resolved_path

        if reset_first:
            self.reset_parameters()

        params_data = data.get("Parameters", [])
        c = self.core._dll
        vals_ptr = c.csmGetParameterValues(self._model_ptr)
        applied_count = 0

        for item in params_data:
            p_id = item.get("Id")
            if not p_id or p_id not in self.parameter_id_map:
                continue
            p_info = self.parameter_id_map[p_id]
            val = float(item.get("Value", 0.0))
            blend = item.get("Blend", 1)
            new_val = _blend_param(vals_ptr[p_info.index], val, blend)
            new_val = max(p_info.min_value, min(p_info.max_value, new_val))
            vals_ptr[p_info.index] = new_val
            p_info.value = new_val
            applied_count += 1

        c.csmUpdateModel(self._model_ptr)
        self._sync_model_state()
        self.current_expression = expr_name

        return {
            "expression": expr_name,
            "source": expr_source,
            "applied_parameters": applied_count,
            "total_expression_parameters": len(params_data),
        }
