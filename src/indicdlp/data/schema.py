"""Explicit, conservative adapters for HF metadata and detection annotations."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass


def unpack(value):
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        return json.loads(value)
    return value


def field(row, path):
    value = row
    for key in path.split("."):
        value = unpack(value)
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return unpack(value)


def first(row, names):
    for name in names:
        value = field(row, name)
        if value is not None:
            return value, name
    return None, None


def normalized(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


DOMAIN_ALIASES = {
    "form": "forms",
    "forms": "forms",
    "fir": "fir",
    "firstinformationreport": "fir",
    "notice": "notices",
    "notices": "notices",
    "actsrules": "acts-and-rules",
    "actsandrules": "acts-and-rules",
    "actandrule": "acts-and-rules",
    "novel": "novels",
    "novels": "novels",
    "textbook": "textbooks",
    "textbooks": "textbooks",
    "magazine": "magazines",
    "magazines": "magazines",
    "manual": "manuals",
    "manuals": "manuals",
    "brochure": "brochures",
    "brochures": "brochures",
    "syllabus": "syllabi",
    "syllabi": "syllabi",
    "newspaper": "newspapers",
    "newspapers": "newspapers",
    "questionpapers": "question-papers",
    "researchpapers": "research-papers",
}

# Verified against the labeled examples in AI4Bharat/IndicDLP's official README:
# https://github.com/AI4Bharat/IndicDLP#-samples-from-the-indicdlp-dataset
INDICDLP_DOMAIN_PREFIXES = {
    "ar": "acts-and-rules",
    "br": "brochures",
    "fm": "forms",
    "mg": "magazines",
    "mn": "manuals",
    "np": "newspapers",
    "nt": "notices",
    "nv": "novels",
    "qp": "question-papers",
    "rp": "research-papers",
    "sy": "syllabi",
    "tb": "textbooks",
}


def indicdlp_filename_metadata(filename):
    """Decode documented domain prefixes and ISO language codes, not source IDs.

    The numeric suffixes are deliberately not interpreted as document/page IDs;
    that relationship has not been established by the HF schema.
    """
    basename = str(filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    match = re.fullmatch(
        r"(?P<domain>[a-z]{2})_(?P<language>[a-z]{2})_\d+_\d+\.(?:png|jpg|jpeg|tif|tiff)",
        basename.lower(),
    )
    if not match:
        return {}
    if (
        match["domain"] not in INDICDLP_DOMAIN_PREFIXES
        or match["language"] not in LANGUAGES
    ):
        return {}
    return {
        "document_type": INDICDLP_DOMAIN_PREFIXES[match["domain"]],
        "language": language(match["language"]),
    }


def domain(value):
    key = normalized(value)
    return DOMAIN_ALIASES.get(key, key)


LANGUAGES = dict(
    zip(
        ["as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"],
        [
            "assamese",
            "bengali",
            "english",
            "gujarati",
            "hindi",
            "kannada",
            "malayalam",
            "marathi",
            "odia",
            "punjabi",
            "tamil",
            "telugu",
        ],
    )
)


def language(value):
    key = normalized(value)
    return LANGUAGES.get(key, "odia" if key == "oriya" else key)


def classlabels(features):
    """Extract named ClassLabel mappings without importing datasets."""
    result = {}

    def walk(value, path):
        if isinstance(value, list):
            for child in value:
                walk(child, path)
            return
        if not isinstance(value, dict):
            return
        if value.get("_type") == "ClassLabel":
            result[path] = value["names"]
        for key, child in value.items():
            if key == "feature":  # Sequence wrapper
                walk(child, path)
            elif not key.startswith("_"):
                walk(child, f"{path}.{key}" if path else key)

    walk(features, "")
    return result


@dataclass
class Adapter:
    names: list[str]
    type_field: str | None = None
    language_field: str | None = None
    filename_field: str | None = None
    annotations_field: str | None = None
    document_id_field: str | None = None
    image_field: str = "image"
    category_id_base: int | None = None
    bbox_format: str = "xywh"

    def decode_label(self, value, path, mappings):
        if isinstance(value, int) and path in mappings:
            if not 0 <= value < len(mappings[path]):
                raise ValueError(f"ClassLabel index out of range in {path}")
            return mappings[path][value]
        return value

    def metadata(self, row, mappings):
        filename, _ = first(
            row,
            [self.filename_field]
            if self.filename_field
            else [
                "file_name",
                "filename",
                "image_path",
                "path",
                f"{self.image_field}.path",
                "metadata.file_name",
                "metadata.filename",
            ],
        )
        encoded_metadata = indicdlp_filename_metadata(filename)
        dtype, path = first(
            row,
            [self.type_field]
            if self.type_field
            else [
                "document_type",
                "doc_type",
                "document_category",
                "doc_category",
                "domain",
                "metadata.document_type",
                "metadata.doc_type",
                "metadata.domain",
            ],
        )
        dtype = self.decode_label(dtype, path, mappings)
        if dtype is None and self.type_field:
            raise ValueError(
                f"Explicit document type field {self.type_field!r} is missing"
            )
        if dtype is None and filename:
            dtype = encoded_metadata.get("document_type")
        if dtype is None and filename:
            # Match complete filename/path tokens, never arbitrary substrings.
            tokens = re.split(r"[^a-z0-9]+", str(filename).lower())
            hits = {domain(t) for t in tokens if normalized(t) in DOMAIN_ALIASES}
            if len(hits) == 1:
                dtype = hits.pop()
            elif re.search(
                r"acts?[ _&/-]+(?:and[ _/-]+)?rules?", str(filename).lower()
            ):
                dtype = "acts-and-rules"
            else:
                for label in ("question-papers", "research-papers"):
                    if re.search(label.replace("-", r"[ _/-]+"), str(filename).lower()):
                        dtype = label
        if dtype is None or not isinstance(dtype, str):
            raise ValueError(
                "Cannot identify document type safely. Run inspect, then set --type-field "
                "to a string/ClassLabel field. Filename inference requires a full domain token "
                "or a documented IndicDLP prefix such as fm_hi_000220_0.png."
            )
        lang, path = first(
            row,
            [self.language_field]
            if self.language_field
            else [
                "language",
                "lang",
                "metadata.language",
                "metadata.lang",
            ],
        )
        lang = self.decode_label(lang, path, mappings)
        if lang is None and self.language_field:
            raise ValueError(
                f"Explicit language field {self.language_field!r} is missing"
            )
        if lang is not None and not isinstance(lang, str):
            raise ValueError(
                "Numeric language metadata requires named HF ClassLabel mappings"
            )
        if lang is None and filename:
            lang = encoded_metadata.get("language")
        if lang is None and filename:
            tokens = re.split(r"[^a-z0-9]+", str(filename).lower())
            hits = {
                language(t) for t in tokens if t in LANGUAGES or t in LANGUAGES.values()
            }
            if len(hits) == 1:
                lang = hits.pop()
        document_id, _ = first(
            row,
            [self.document_id_field]
            if self.document_id_field
            else [
                "document_id",
                "doc_id",
                "source_document_id",
                "metadata.document_id",
                "metadata.doc_id",
            ],
        )
        if self.document_id_field and document_id is None:
            raise ValueError(
                f"Explicit document ID field {self.document_id_field!r} is missing"
            )
        if document_id is not None and not isinstance(document_id, (str, int)):
            raise ValueError("Source document ID must be a string or integer")
        return {
            "document_type": domain(dtype),
            "language": language(lang) if lang else None,
            "filename": str(filename) if filename is not None else None,
            "document_id": str(document_id) if document_id is not None else None,
        }

    def annotations(self, row, width, height, mappings):
        value, path = first(
            row,
            [self.annotations_field]
            if self.annotations_field
            else [
                "annotations",
                "objects",
                "labels",
                "label",
                "annotation",
                "yolo_labels",
            ],
        )
        if (
            value is None
            and not self.annotations_field
            and field(row, "bboxes") is not None
            and field(row, "category_ids") is not None
        ):
            # Native ai4bharat/indicdlp Parquet schema, confirmed from authenticated
            # metadata inspection at revision 7f23f6d8569dd6336534dd9700259b9843ecab5e.
            value = {
                "bbox": field(row, "bboxes"),
                "category_ids": field(row, "category_ids"),
            }
            path = ""
        if value is None:
            raise ValueError(
                "No annotations found; set --annotations-field after inspect"
            )
        if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
            value = "\n".join(value)
        if isinstance(value, str):
            # Native YOLO text always has zero-based class IDs and normalized xywh.
            lines = []
            for text in value.splitlines():
                if not text.strip():
                    continue
                parts = text.split()
                if len(parts) != 5:
                    raise ValueError("YOLO text annotation must have five columns")
                cid = self.class_id(parts[0], None, {}, zero_based=True)
                box = list(map(float, parts[1:]))
                self.validate_yolo(box)
                lines.append(self.line(cid, box))
            return lines, 0
        categories = {}
        if isinstance(value, dict) and "categories" in value:
            categories = {c["id"]: c["name"] for c in value["categories"]}
        if isinstance(value, dict) and "annotations" in value:
            value = value["annotations"]
            path += ".annotations"
        if isinstance(value, dict):
            boxes, box_key = first(value, ["bbox", "bboxes", "boxes"])
            labels, label_key = first(
                value,
                [
                    "category_id",
                    "category_ids",
                    "category",
                    "class_id",
                    "labels",
                    "label",
                ],
            )
            if boxes is None or labels is None:
                raise ValueError(
                    "Unknown annotation dictionary; expected bbox and category_id"
                )
            if boxes and isinstance(boxes[0], (int, float)):
                value = [{"bbox": boxes, "category_id": labels}]
            else:
                if len(boxes) != len(labels):
                    raise ValueError(
                        "Annotation boxes and labels have different lengths"
                    )
                value = [
                    {"bbox": box, "category_id": label}
                    for box, label in zip(boxes, labels)
                ]
            label_path = f"{path}.{label_key}" if path else label_key
        else:
            label_path = None
        if not isinstance(value, list):
            raise ValueError("Expected COCO records, objects dictionary, or YOLO text")
        lines, dropped = [], 0
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("Annotation records must be dictionaries")
            if item.get("iscrowd", 0) or item.get("ignore", 0):
                raise ValueError(
                    "Crowd/ignore boxes require an explicit annotation policy"
                )
            box, _ = first(item, ["bbox", "box", "bboxes"])
            label, key = first(
                item, ["category_id", "category", "class_id", "label", "class_name"]
            )
            cid = self.class_id(
                label, label_path or f"{path}.{key}", mappings, categories
            )
            converted = self.convert_box(box, width, height)
            if converted is None:
                dropped += 1
            else:
                lines.append(self.line(cid, converted))
        return lines, dropped

    def class_id(self, value, path, mappings, categories=None, zero_based=False):
        value = self.decode_label(value, path, mappings)
        if categories and value in categories:
            value = categories[value]
        if isinstance(value, str) and not value.lstrip("-").isdigit():
            lookup = {normalized(n): i for i, n in enumerate(self.names)}
            if normalized(value) not in lookup:
                raise ValueError(f"Unknown layout class {value!r}")
            return lookup[normalized(value)]
        if self.category_id_base is None and not zero_based:
            raise ValueError(
                "Numeric region IDs have no ClassLabel/category names. Specify "
                "--category-id-base 0 or 1 after verifying the source convention; never guess."
            )
        if isinstance(value, bool) or value is None or float(value) != int(value):
            raise ValueError(f"Invalid class ID {value!r}")
        cid = int(value) - (0 if zero_based else self.category_id_base)
        if not 0 <= cid < len(self.names):
            raise ValueError(f"Class ID out of range: {cid}")
        return cid

    @staticmethod
    def validate_yolo(box):
        if len(box) != 4 or not all(math.isfinite(x) for x in box):
            raise ValueError("Invalid box coordinates")
        x, y, w, h = box
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(
                "YOLO coordinates must be normalized, with positive width/height"
            )
        if (
            x - w / 2 < -1e-6
            or y - h / 2 < -1e-6
            or x + w / 2 > 1.000001
            or y + h / 2 > 1.000001
        ):
            raise ValueError("YOLO box extends outside image")

    def convert_box(self, box, width, height):
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("bbox must contain four coordinates")
        x, y, a, b = map(float, box)
        if not all(math.isfinite(v) for v in (x, y, a, b)):
            raise ValueError("Non-finite box coordinate")
        if self.bbox_format == "yolo":
            result = [x, y, a, b]
            self.validate_yolo(result)
            return result
        x2, y2 = (x + a, y + b) if self.bbox_format == "xywh" else (a, b)
        x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return [
            (x1 + x2) / (2 * width),
            (y1 + y2) / (2 * height),
            (x2 - x1) / width,
            (y2 - y1) / height,
        ]

    @staticmethod
    def line(cid, box):
        return f"{cid} " + " ".join(f"{v:.8f}" for v in box)
