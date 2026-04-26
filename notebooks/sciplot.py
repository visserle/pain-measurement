import base64
import os
from copy import deepcopy
from pathlib import Path

from dotenv import load_dotenv
from lxml import etree
from PIL import Image

load_dotenv()
FIGURE_DIR = Path(os.getenv("SCI_DATA_FIGURE_DIR"))

SVG_NS = "http://www.w3.org/2000/svg"
_RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def _viewbox_dims(path: Path) -> tuple[float, float]:
    if path.suffix.lower() in _RASTER_SUFFIXES:
        with Image.open(path) as img:
            return float(img.width), float(img.height)
    root = etree.parse(str(path)).getroot()
    vb = root.get("viewBox")
    if vb:
        _, _, w, h = vb.split()
        return float(w), float(h)
    return float(root.get("width", 0)), float(root.get("height", 0))


def compose_panel_figure(
    output_path: Path | str,
    row1: Path,
    row2_left: Path,
    row2_right: Path,
    gap: int = 16,
    label_size: int = 18,
) -> None:
    """Compose three SVGs/images: one full-width top row, two side-by-side bottom row.

    Scales both bottom panels to the same height so they fill the canvas width.
    Adds lowercase alphabetical panel labels (a, b, c).
    Supports SVG and raster images (PNG, JPEG, etc.) for any panel.
    """
    w_a, h_a = _viewbox_dims(row1)
    w_b, h_b = _viewbox_dims(row2_left)
    w_c, h_c = _viewbox_dims(row2_right)

    canvas_w = int(w_a)

    # Scale B and C to equal height so (wB + gap + wC) == canvas_w
    ar_b, ar_c = w_b / h_b, w_c / h_c
    h2 = (canvas_w - gap) / (ar_b + ar_c)
    w_panel_b = round(h2 * ar_b)
    w_panel_c = canvas_w - gap - w_panel_b
    h_row2 = round(h2)

    canvas_h = round(h_a) + gap + h_row2

    nsmap = {None: SVG_NS}
    root_svg = etree.Element(f"{{{SVG_NS}}}svg", nsmap=nsmap)
    root_svg.set("width", str(canvas_w))
    root_svg.set("height", str(canvas_h))
    root_svg.set("viewBox", f"0 0 {canvas_w} {canvas_h}")

    bg = etree.SubElement(root_svg, f"{{{SVG_NS}}}rect")
    bg.set("width", "100%")
    bg.set("height", "100%")
    bg.set("fill", "white")

    y2 = round(h_a) + gap
    panels = [
        # (label, path, x, y, w, h, vb_w, vb_h)
        ("a", row1, 0, 0, canvas_w, round(h_a), int(w_a), int(h_a)),
        ("b", row2_left, 0, y2, w_panel_b, h_row2, int(w_b), int(h_b)),
        ("c", row2_right, w_panel_b + gap, y2, w_panel_c, h_row2, int(w_c), int(h_c)),
    ]

    for label, path, x, y, w, h, vb_w, vb_h in panels:
        if path.suffix.lower() in _RASTER_SUFFIXES:
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            encoded = base64.b64encode(path.read_bytes()).decode()
            img_elem = etree.SubElement(root_svg, f"{{{SVG_NS}}}image")
            img_elem.set("x", str(x))
            img_elem.set("y", str(y))
            img_elem.set("width", str(w))
            img_elem.set("height", str(h))
            img_elem.set("href", f"data:{mime};base64,{encoded}")
            img_elem.set("preserveAspectRatio", "xMidYMid meet")
        else:
            src_root = etree.parse(str(path)).getroot()
            child_svg = etree.SubElement(root_svg, f"{{{SVG_NS}}}svg")
            child_svg.set("x", str(x))
            child_svg.set("y", str(y))
            child_svg.set("width", str(w))
            child_svg.set("height", str(h))
            child_svg.set("viewBox", f"0 0 {vb_w} {vb_h}")
            child_svg.set("preserveAspectRatio", "xMidYMid meet")
            for child in src_root:
                child_svg.append(deepcopy(child))

        text = etree.SubElement(root_svg, f"{{{SVG_NS}}}text")
        text.set("x", str(x))
        text.set("y", str(y + label_size - 3))
        text.set("font-family", "Arial, Helvetica, sans-serif")
        text.set("font-size", str(label_size))
        text.set("font-weight", "bold")
        text.text = label

    etree.ElementTree(root_svg).write(
        str(output_path),
        pretty_print=True,
        xml_declaration=True,
        encoding="unicode",
    )
    print(f"Saved → {output_path}")


compose_panel_figure(
    output_path=FIGURE_DIR / "figure_1.svg",
    row1=FIGURE_DIR / "stimulus_with_labels_ci.svg",
    row2_left=FIGURE_DIR / "stimulus_seed_grid.svg",
    row2_right=FIGURE_DIR / "forearm_sci_dat.png",
)
