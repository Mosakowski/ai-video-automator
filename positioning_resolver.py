from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


VisualBBox = Tuple[int, int, int, int]  # x, y, w, h


def extract_visual_bbox_from_rgba(image_rgba: np.ndarray, alpha_threshold: int = 8) -> VisualBBox:
    """
    Return the visual bounding box using alpha channel coverage.
    If image has no alpha or is fully transparent, fall back to full canvas.
    """
    if image_rgba is None or image_rgba.size == 0:
        return (0, 0, 1, 1)

    h, w = image_rgba.shape[:2]
    if image_rgba.ndim < 3 or image_rgba.shape[2] < 4:
        return (0, 0, w, h)

    alpha = image_rgba[:, :, 3]
    ys, xs = np.where(alpha > alpha_threshold)

    if xs.size == 0 or ys.size == 0:
        return (0, 0, w, h)

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)


def parse_position_spec(position_spec: str) -> Dict[str, object]:
    """
    Supported formats:
    - XY:x,y
    - GRID:<token>
    - legacy token without prefix (e.g. Center, Bottom-Right)
    """
    spec = (position_spec or "").strip()

    if spec.startswith("XY:"):
        raw = spec.replace("XY:", "", 1)
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 2:
            return {"mode": "XY", "xy": (0, 0), "token": None}
        try:
            return {"mode": "XY", "xy": (int(float(parts[0])), int(float(parts[1]))), "token": None}
        except Exception:
            return {"mode": "XY", "xy": (0, 0), "token": None}

    if spec.startswith("GRID:"):
        token = spec.replace("GRID:", "", 1).strip()
        return {"mode": "GRID", "xy": None, "token": token or "Center"}

    return {"mode": "GRID", "xy": None, "token": spec or "Center"}


def _resolve_grid_token(token: str) -> Tuple[str, str]:
    t = (token or "Center").strip()

    col = "Center"
    row = "Center"

    if "Left" in t and "Center-Left" not in t:
        col = "Left"
    if "Right" in t and "Center-Right" not in t:
        col = "Right"
    if "Center-Left" in t:
        col = "Left"
    if "Center-Right" in t:
        col = "Right"
    if "Center" in t and "Left" not in t and "Right" not in t:
        col = "Center"

    if "Top" in t:
        row = "Top"
    elif "Upper-Middle" in t:
        row = "Upper-Middle"
    elif "Lower-Middle" in t:
        row = "Lower-Middle"
    elif "Bottom" in t:
        row = "Bottom"
    elif "Center" in t:
        row = "Center"

    return col, row


def resolve_header_top_left(
    position_spec: str,
    target_size: Tuple[int, int],
    canvas_size: Tuple[int, int],
    visual_bbox: Optional[VisualBBox] = None,
    grid_margin: int = 60,
    xy_scale: Tuple[float, float] = (1.0, 1.0),
) -> Tuple[int, int]:
    """
    Return top-left coordinates for the full header canvas.
    GRID mode aligns by visual bbox (optical anchor).
    XY mode treats input as top-left canvas coordinates.
    """
    target_w, target_h = target_size
    canvas_w, canvas_h = canvas_size

    if visual_bbox is None:
        visual_bbox = (0, 0, canvas_w, canvas_h)

    vb_x, vb_y, vb_w, vb_h = visual_bbox

    parsed = parse_position_spec(position_spec)
    mode = parsed["mode"]

    if mode == "XY":
        x_raw, y_raw = parsed["xy"]
        sx, sy = xy_scale
        return int(round(x_raw * sx)), int(round(y_raw * sy))

    token = parsed["token"]
    col, row = _resolve_grid_token(token)

    if col == "Left":
        visual_left = grid_margin
        x = visual_left - vb_x
    elif col == "Right":
        visual_right = target_w - grid_margin
        x = visual_right - (vb_x + vb_w)
    else:
        visual_cx = target_w * 0.5
        x = int(round(visual_cx - (vb_x + vb_w * 0.5)))

    if row == "Top":
        visual_top = grid_margin
        y = visual_top - vb_y
    elif row == "Upper-Middle":
        visual_cy = target_h * 0.25
        y = int(round(visual_cy - (vb_y + vb_h * 0.5)))
    elif row == "Lower-Middle":
        visual_cy = target_h * 0.75
        y = int(round(visual_cy - (vb_y + vb_h * 0.5)))
    elif row == "Bottom":
        visual_bottom = target_h - grid_margin
        y = visual_bottom - (vb_y + vb_h)
    else:
        visual_cy = target_h * 0.5
        y = int(round(visual_cy - (vb_y + vb_h * 0.5)))

    return int(x), int(y)
