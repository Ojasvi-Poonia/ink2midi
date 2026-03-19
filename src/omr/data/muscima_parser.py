"""Parse MUSCIMA++ v2 XML annotations into unified format."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from omr.utils.logging import get_logger

logger = get_logger("data.muscima_parser")


@dataclass
class SymbolAnnotation:
    """Unified symbol annotation across datasets."""

    symbol_id: int
    class_name: str
    class_id: int
    bbox: tuple[float, float, float, float]  # (x_center, y_center, w, h) normalized
    bbox_abs: tuple[int, int, int, int]  # (top, left, width, height) absolute pixels
    mask: Optional[np.ndarray] = field(default=None, repr=False)
    image_id: str = ""
    confidence: float = 1.0


@dataclass
class RelationshipAnnotation:
    """Edge in the notation graph."""

    source_id: int
    target_id: int
    source_class: str
    target_class: str
    relationship_type: str  # 'outlink' or 'inlink'


@dataclass
class DocumentAnnotation:
    """All annotations for a single page/image."""

    image_id: str
    image_path: str
    image_width: int
    image_height: int
    symbols: list[SymbolAnnotation] = field(default_factory=list)
    relationships: list[RelationshipAnnotation] = field(default_factory=list)


# Map MUSCIMA++ v2 class names to our 26 unified classes
# Classes not in this map are ignored (rare/complex symbols)
CLASS_MAP = {
    # Noteheads
    "noteheadFull": "notehead_filled",
    "noteheadHalf": "notehead_half",
    "noteheadWhole": "notehead_whole",
    "noteheadFullSmall": "notehead_filled",
    "noteheadHalfSmall": "notehead_half",
    # Stems
    "stem": "stem",
    # Beams
    "beam": "beam",
    # Flags
    "flag8thUp": "flag_eighth_up",
    "flag8thDown": "flag_eighth_down",
    "flag16thUp": "flag_sixteenth_up",
    "flag16thDown": "flag_sixteenth_down",
    # Rests
    "restWhole": "rest_whole",
    "restHalf": "rest_half",
    "restQuarter": "rest_quarter",
    "rest8th": "rest_eighth",
    "rest16th": "rest_sixteenth",
    # Accidentals
    "accidentalSharp": "sharp",
    "accidentalFlat": "flat",
    "accidentalNatural": "natural",
    "accidentalDoubleSharp": "double_sharp",
    # Clefs (MUSCIMA++ v2 uses gClef/fClef/cClef)
    "gClef": "treble_clef",
    "fClef": "bass_clef",
    "cClef": "alto_clef",
    # Time signatures
    "timeSignature": "time_sig_digit",
    "numeral0": "time_sig_digit",
    "numeral1": "time_sig_digit",
    "numeral2": "time_sig_digit",
    "numeral3": "time_sig_digit",
    "numeral4": "time_sig_digit",
    "numeral5": "time_sig_digit",
    "numeral6": "time_sig_digit",
    "numeral7": "time_sig_digit",
    "numeral8": "time_sig_digit",
    "numeral9": "time_sig_digit",
    # Barlines
    "barline": "bar_line",
    "barlineHeavy": "double_bar_line",
    # Dots
    "augmentationDot": "dot",
    # Ties and slurs
    "tie": "tie_slur",
    "slur": "tie_slur",
}

# Unified class name → numeric ID
CLASS_NAME_TO_ID = {
    "notehead_filled": 0,
    "notehead_half": 1,
    "notehead_whole": 2,
    "stem": 3,
    "beam": 4,
    "flag_eighth_up": 5,
    "flag_eighth_down": 6,
    "flag_sixteenth_up": 7,
    "flag_sixteenth_down": 8,
    "rest_whole": 9,
    "rest_half": 10,
    "rest_quarter": 11,
    "rest_eighth": 12,
    "rest_sixteenth": 13,
    "sharp": 14,
    "flat": 15,
    "natural": 16,
    "double_sharp": 17,
    "treble_clef": 18,
    "bass_clef": 19,
    "alto_clef": 20,
    "time_sig_digit": 21,
    "bar_line": 22,
    "double_bar_line": 23,
    "dot": 24,
    "tie_slur": 25,
}


class MUSCIMAParser:
    """Parse MUSCIMA++ v2 XML annotation files."""

    def __init__(self):
        self.class_map = CLASS_MAP
        self.class_to_id = CLASS_NAME_TO_ID
        self._id_map: dict[str, int] = {}  # XML id → local int id

    def parse_document(
        self,
        xml_path: str | Path,
        image_width: int,
        image_height: int,
        image_path: str = "",
    ) -> DocumentAnnotation:
        """Parse one MUSCIMA++ v2 XML file.

        Args:
            xml_path: Path to the annotation XML.
            image_width: Width of the source image in pixels.
            image_height: Height of the source image in pixels.
            image_path: Path to the corresponding image file.

        Returns:
            DocumentAnnotation with all symbols and relationships.
        """
        xml_path = Path(xml_path)
        self._id_map = {}  # Reset for each document to avoid ID leaks
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Determine namespace if present
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        doc = DocumentAnnotation(
            image_id=xml_path.stem,
            image_path=str(image_path),
            image_width=image_width,
            image_height=image_height,
        )

        # First pass: parse all nodes and build ID map
        nodes_by_id: dict[str, dict] = {}
        for node_elem in root.iter(f"{ns}Node"):
            node_data = self._parse_node(node_elem, ns, image_width, image_height)
            if node_data:
                nodes_by_id[node_data["xml_id"]] = node_data

        # Build symbol list from mapped classes
        for xml_id, node_data in nodes_by_id.items():
            if node_data["symbol"] is not None:
                doc.symbols.append(node_data["symbol"])

        # Second pass: parse relationships (outlinks)
        for node_elem in root.iter(f"{ns}Node"):
            xml_id = self._get_text(node_elem, f"{ns}Id")
            if xml_id not in nodes_by_id:
                continue

            outlinks_elem = node_elem.find(f"{ns}Outlinks")
            if outlinks_elem is not None and outlinks_elem.text:
                target_ids = [t.strip() for t in outlinks_elem.text.split()]
                for target_id in target_ids:
                    if target_id in nodes_by_id:
                        src_data = nodes_by_id[xml_id]
                        tgt_data = nodes_by_id[target_id]
                        if src_data["symbol"] is not None and tgt_data["symbol"] is not None:
                            rel = RelationshipAnnotation(
                                source_id=src_data["symbol"].symbol_id,
                                target_id=tgt_data["symbol"].symbol_id,
                                source_class=src_data["symbol"].class_name,
                                target_class=tgt_data["symbol"].class_name,
                                relationship_type="outlink",
                            )
                            doc.relationships.append(rel)

        logger.debug(
            f"Parsed {xml_path.name}: {len(doc.symbols)} symbols, "
            f"{len(doc.relationships)} relationships"
        )
        return doc

    def _parse_node(
        self, node_elem, ns: str, img_w: int, img_h: int
    ) -> dict | None:
        """Parse a single Node element."""
        xml_id = self._get_text(node_elem, f"{ns}Id")
        class_name_raw = self._get_text(node_elem, f"{ns}ClassName")

        if not xml_id or not class_name_raw:
            return None

        # Map to unified class
        unified_name = self.class_map.get(class_name_raw)
        if unified_name is None:
            # Class not in our target set, still track for relationships
            return {
                "xml_id": xml_id,
                "symbol": None,
                "class_raw": class_name_raw,
            }

        class_id = self.class_to_id[unified_name]

        # Parse bounding box
        top = int(self._get_text(node_elem, f"{ns}Top") or 0)
        left = int(self._get_text(node_elem, f"{ns}Left") or 0)
        width = int(self._get_text(node_elem, f"{ns}Width") or 0)
        height = int(self._get_text(node_elem, f"{ns}Height") or 0)

        # Absolute bbox
        bbox_abs = (top, left, width, height)

        # Normalized bbox (YOLO format: x_center, y_center, w, h)
        x_center = (left + width / 2) / img_w
        y_center = (top + height / 2) / img_h
        w_norm = width / img_w
        h_norm = height / img_h
        bbox_norm = (x_center, y_center, w_norm, h_norm)

        # Parse mask if present
        mask = self._parse_mask(node_elem, ns, width, height)

        # Generate numeric symbol ID
        symbol_id = len(self._id_map)
        self._id_map[xml_id] = symbol_id

        symbol = SymbolAnnotation(
            symbol_id=symbol_id,
            class_name=unified_name,
            class_id=class_id,
            bbox=bbox_norm,
            bbox_abs=bbox_abs,
            mask=mask,
            confidence=1.0,
        )

        return {
            "xml_id": xml_id,
            "symbol": symbol,
            "class_raw": class_name_raw,
        }

    def _parse_mask(self, node_elem, ns: str, width: int, height: int) -> np.ndarray | None:
        """Parse pixel mask from the Mask element."""
        mask_elem = node_elem.find(f"{ns}Mask")
        if mask_elem is None or not mask_elem.text:
            return None

        try:
            pixels = mask_elem.text.strip().split()
            mask = np.zeros((height, width), dtype=np.uint8)
            for i, row_str in enumerate(pixels):
                if i >= height:
                    break
                for j, ch in enumerate(row_str):
                    if j >= width:
                        break
                    if ch == "1":
                        mask[i, j] = 1
            return mask
        except (ValueError, IndexError):
            return None

    def _get_text(self, elem, tag: str) -> str | None:
        """Get text content of a child element."""
        child = elem.find(tag)
        if child is not None and child.text:
            return child.text.strip()
        return None

    def parse_directory(
        self,
        annotations_dir: str | Path,
        images_dir: str | Path,
        default_width: int = 3480,
        default_height: int = 4800,
    ) -> list[DocumentAnnotation]:
        """Parse all XML files in a directory.

        Args:
            annotations_dir: Directory containing MUSCIMA++ XML files.
            images_dir: Directory containing corresponding images.
            default_width: Default image width if not determinable.
            default_height: Default image height if not determinable.

        Returns:
            List of DocumentAnnotation objects.
        """
        annotations_dir = Path(annotations_dir)
        images_dir = Path(images_dir)
        documents = []

        xml_files = sorted(annotations_dir.glob("*.xml"))
        logger.info(f"Found {len(xml_files)} XML annotation files")

        for xml_path in xml_files:
            # Reset ID map for each document
            self._id_map = {}

            # Try to find corresponding image
            image_path = self._find_image(xml_path.stem, images_dir)
            img_w, img_h = default_width, default_height

            if image_path:
                from PIL import Image

                with Image.open(image_path) as img:
                    img_w, img_h = img.size

            doc = self.parse_document(xml_path, img_w, img_h, str(image_path or ""))
            documents.append(doc)

        logger.info(
            f"Parsed {len(documents)} documents with "
            f"{sum(len(d.symbols) for d in documents)} total symbols"
        )
        return documents

    def _find_image(self, stem: str, images_dir: Path) -> Path | None:
        """Find the corresponding image file for an annotation.

        MUSCIMA++ annotations are named like:
            CVC-MUSCIMA_W-XX_N-YY_D-ideal
        But CVC-MUSCIMA images use a different naming convention:
            CvcMuscima-Distortions/ideal/w-XX/symbol/pYYY.png
        where XX is the writer number and YYY is the page number (zero-padded to 3 digits).
        """
        # Direct match first (works if images are renamed to match annotations)
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate

        # Try CVC-MUSCIMA naming convention:
        # Extract writer (W-XX) and page (N-YY) from annotation stem
        import re
        match = re.match(r"CVC-MUSCIMA_W-(\d+)_N-(\d+)_D-(\w+)", stem)
        if match:
            writer_num = match.group(1)  # e.g. "01"
            page_num = int(match.group(2))  # e.g. 10
            distortion = match.group(3)  # e.g. "ideal"
            page_str = f"p{page_num:03d}"  # e.g. "p010"

            # Search known CVC-MUSCIMA directory structures
            cvc_patterns = [
                f"CvcMuscima-Distortions/{distortion}/w-{writer_num}/symbol/{page_str}.png",
                f"CvcMuscima-Distortions/{distortion}/w-{writer_num}/image/{page_str}.png",
                f"CvcMuscima-Distortions/{distortion}/w-{writer_num}/{page_str}.png",
                f"CVC-MUSCIMA_SR/w-{writer_num}/image/{page_str}.png",
                f"CVC-MUSCIMA_SR/w-{writer_num}/{page_str}.png",
                f"w-{writer_num}/symbol/{page_str}.png",
                f"w-{writer_num}/image/{page_str}.png",
                f"w-{writer_num}/{page_str}.png",
            ]
            for pattern in cvc_patterns:
                candidate = images_dir / pattern
                if candidate.exists():
                    return candidate

            # Recursive search for the page file in writer directory
            for candidate in images_dir.rglob(f"w-{writer_num}/*/{page_str}.png"):
                return candidate
            for candidate in images_dir.rglob(f"w-{writer_num}/{page_str}.png"):
                return candidate

        # Generic recursive search as last resort
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            candidates = list(images_dir.rglob(f"{stem}{ext}"))
            if candidates:
                return candidates[0]
        return None
