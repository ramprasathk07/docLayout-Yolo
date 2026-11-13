"""Local evaluation reports contain metrics only, never document images or tokens."""

import csv
import io
from pathlib import Path

from ..data.prepare import atomic, json_write


def csv_text(rows):
    result = io.StringIO(newline="")
    csv.writer(result).writerows(rows)
    return result.getvalue()


def write_report(output, report):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    json_write(output / "metrics.json", report)
    columns = [
        "class_id",
        "name",
        "targets",
        "evaluated",
        "precision",
        "recall",
        "f1",
        "map50",
        "map75",
        "map50_95",
    ]
    rows = [columns] + [
        [cid] + [values.get(k) for k in columns[1:]]
        for cid, values in report["per_class"].items()
    ]
    atomic(output / "per_class.csv", csv_text(rows))
    matrix = report.get("confusion_matrix")
    if matrix:
        atomic(
            output / "confusion_matrix.csv",
            csv_text(
                [["predicted / actual", *matrix["labels"]]]
                + [
                    [label, *row]
                    for label, row in zip(matrix["labels"], matrix["counts"])
                ]
            ),
        )
    json_write(output / "curves.json", report.get("curves", []))
    title = report.get("split", "validation")
    lines = [
        f"# {title.title()} detection metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    lines += [f"| {key} | {value:.6f} |" for key, value in report["aggregate"].items()]
    lines += [
        "",
        f"Images: {report.get('images')}",
        "",
        report["operating_point"],
        "",
        report["missing_classes"],
        "",
        "Per-class results: `per_class.csv`. Raw confusion matrix and full curves: `metrics.json`.",
        "",
    ]
    atomic(output / "summary.md", "\n".join(lines))
    return output
