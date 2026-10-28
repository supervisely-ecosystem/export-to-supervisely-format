"""add_additional_label_fields must leave an archive whose annotations match meta.json.

It renames classes/tags whose names have non-printable characters or surrounding
whitespace (sanitize_name_if_needed) and adds curved edges for images with camera
calibration in their meta.
"""

import os

import numpy as np
import supervisely as sly

from conftest import make_api

CAR = "car "  # trailing space: remove_non_printable() strips it -> "car"
POLY = {"exterior": [[10, 10], [40, 10], [40, 40]], "interior": []}


def _label(points=POLY):
    return {"classTitle": CAR, "geometryType": "polygon", "points": points, "tags": []}


def _write_project(root, items):
    """items: {name: (image_meta or None, [label json])} in one dataset 'ds'."""
    meta = sly.ProjectMeta(obj_classes=[sly.ObjClass(CAR, sly.Polygon)])
    os.makedirs(root, exist_ok=True)
    sly.json.dump_json_file(meta.to_json(), os.path.join(root, "meta.json"))
    ds = os.path.join(root, "ds")
    for sub in ("img", "ann", "meta"):
        os.makedirs(os.path.join(ds, sub), exist_ok=True)
    for name, (image_meta, labels) in items.items():
        sly.image.write(os.path.join(ds, "img", name), np.zeros((50, 50, 3), dtype=np.uint8))
        ann = {"description": "", "size": {"height": 50, "width": 50}, "tags": [], "objects": labels}
        sly.json.dump_json_file(ann, os.path.join(ds, "ann", f"{name}.json"))
        if image_meta is not None:
            sly.json.dump_json_file(image_meta, os.path.join(ds, "meta", f"{name}.json"))


def _read_all(root):
    """What an importer does: every annotation must deserialize against meta.json."""
    project = sly.Project(root, sly.OpenMode.READ)
    anns = {}
    for dataset in project.datasets:
        for name in dataset.get_items_names():
            anns[name] = dataset.get_ann(name, project.meta)
    return project.meta, anns


def _calibration(fx):
    return {"calibration": {"intrinsic": {"fx": fx, "fy": 100.0, "cx": 25.0, "cy": 25.0}}}


def test_item_meta_without_calibration_still_gets_sanitized_names(load_main, tmp_path):
    main = load_main(make_api())
    root = str(tmp_path / "project")
    _write_project(
        root,
        {
            "a.png": ({"source": "camera-3"}, [_label()]),  # meta, but no calibration
            "b.png": (None, [_label()]),
        },
    )
    main.add_additional_label_fields(root)
    meta, anns = _read_all(root)
    assert [c.name for c in meta.obj_classes] == ["car"]
    assert {n: [lb.obj_class.name for lb in a.labels] for n, a in anns.items()} == {
        "a.png": ["car"],
        "b.png": ["car"],
    }


def test_bad_calibration_on_one_item_does_not_abort_the_rest(load_main, tmp_path):
    main = load_main(make_api())
    root = str(tmp_path / "project")
    _write_project(
        root,
        {
            "a.png": (_calibration(fx=0.0), [_label()]),  # singular K: curved edges fail
            "b.png": (_calibration(fx=100.0), [_label()]),
            "c.png": (None, [_label()]),
        },
    )
    main.add_additional_label_fields(root)
    meta, anns = _read_all(root)
    assert {n: [lb.obj_class.name for lb in a.labels] for n, a in anns.items()} == {
        "a.png": ["car"],
        "b.png": ["car"],
        "c.png": ["car"],
    }
    raw_b = sly.json.load_json_file(os.path.join(root, "ds", "ann", "b.png.json"))
    assert "_curved_cylindrical_edges" in raw_b["objects"][0]


def test_sanitizing_into_an_existing_name_keeps_both_classes(load_main, tmp_path):
    main = load_main(make_api())
    root = str(tmp_path / "project")
    meta = sly.ProjectMeta(obj_classes=[sly.ObjClass(CAR, sly.Polygon), sly.ObjClass("car", sly.Polygon)])
    _write_project(root, {"a.png": (None, [_label()])})
    sly.json.dump_json_file(meta.to_json(), os.path.join(root, "meta.json"))
    main.add_additional_label_fields(root)
    meta, anns = _read_all(root)
    assert sorted(c.name for c in meta.obj_classes) == ["car", "car "]
    assert [lb.obj_class.name for lb in anns["a.png"].labels] == [CAR]
