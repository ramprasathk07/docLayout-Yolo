"""Convert the upstream detection metric objects into stable, JSON-safe reports."""

from __future__ import annotations

import math
import statistics


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def scalars(values):
    return {
        str(k): n
        for k, value in (values or {}).items()
        if (n := number(value)) is not None
    }


def plain(value):
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    return number(value)


def element(values, index):
    try:
        return number(values[index])
    except (IndexError, TypeError):
        return None


def snapshot(metrics, validator=None, diagnostics=True):
    box = getattr(metrics, "box", None)
    names = getattr(metrics, "names", None) or getattr(validator, "names", {})
    if isinstance(names, list):
        names = dict(enumerate(names))
    names = {int(k): v for k, v in names.items()}
    indices = [int(i) for i in getattr(box, "ap_class_index", [])]
    if not names:
        names = {cid: str(cid) for cid in indices}
    counts = getattr(validator, "nt_per_class", None)
    per_class = {}
    for cid, name in sorted(names.items()):
        support = element(counts, cid)
        row = {
            "name": name,
            "targets": int(support) if support is not None else None,
            "precision": None,
            "recall": None,
            "f1": None,
            "map50": None,
            "map75": None,
            "map50_95": None,
            "ap_by_iou": None,
        }
        if cid in indices:
            i = indices.index(cid)
            for key, attr in (
                ("precision", "p"),
                ("recall", "r"),
                ("f1", "f1"),
                ("map50", "ap50"),
                ("map50_95", "ap"),
            ):
                row[key] = element(getattr(box, attr, []), i)
            all_ap = getattr(box, "all_ap", [])
            if len(all_ap) > i:
                row["ap_by_iou"] = plain(all_ap[i])
                row["map75"] = element(all_ap[i], 5)
        elif support is not None and support > 0:
            # Present targets with no matching predictions must contribute zero AP.
            row.update(
                {
                    k: 0.0
                    for k in ("precision", "recall", "f1", "map50", "map75", "map50_95")
                }
            )
            row["ap_by_iou"] = [0.0] * 10
        row["evaluated"] = cid in indices or bool(support)
        per_class[str(cid)] = row
    aggregate = scalars(getattr(metrics, "results_dict", {}))
    f1 = [row["f1"] for row in per_class.values() if row["f1"] is not None]
    ap75 = [row["map75"] for row in per_class.values() if row["map75"] is not None]
    if f1:
        aggregate["metrics/macro_F1(B)"] = statistics.mean(f1)
    if ap75:
        aggregate["metrics/mAP75(B)"] = statistics.mean(ap75)
    report = {
        "aggregate": aggregate,
        "per_class": per_class,
        "speed_ms_per_image": scalars(getattr(metrics, "speed", {})),
        "images": getattr(validator, "seen", None),
        "iou_thresholds": [round(0.5 + 0.05 * i, 2) for i in range(10)],
        "operating_point": "Upstream precision/recall/F1 use the confidence maximizing mean F1; AP integrates the PR curve.",
        "missing_classes": "Classes without validation targets have undefined AP (null), not an invented zero.",
        "confusion_matrix": None,
        "curves": [],
    }
    if diagnostics:
        confusion = getattr(metrics, "confusion_matrix", None) or getattr(
            validator, "confusion_matrix", None
        )
        matrix = getattr(confusion, "matrix", None)
        if matrix is not None:
            report["confusion_matrix"] = {
                "labels": [names[c] for c in sorted(names)] + ["background"],
                "counts": plain(matrix),
                "rows": "predicted",
                "columns": "actual",
                "confidence_threshold": number(getattr(confusion, "conf", None)),
                "iou_threshold": number(getattr(confusion, "iou_thres", None)),
            }
        for name, values in zip(
            getattr(metrics, "curves", []), getattr(metrics, "curves_results", [])
        ):
            x, y, xlabel, ylabel = values
            report["curves"].append(
                {
                    "name": name,
                    "x": plain(x),
                    "y": plain(y),
                    "x_label": xlabel,
                    "y_label": ylabel,
                    "class_ids": indices,
                }
            )
    return plain(report)
