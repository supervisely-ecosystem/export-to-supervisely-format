"""An export must never write to the project it exports.

Old behaviour: when the project meta failed to deserialize, the app "repaired" it on the
server with api.project.update_meta (disabling multi-view, or replacing the meta with the
settings-less one) before downloading.
"""

import os

import supervisely as sly

from conftest import FakeProjectApi, make_api

GOOD_META = {"classes": [{"title": "car", "shape": "rectangle", "color": "#FF0000"}], "tags": []}


def _broken(settings):
    meta = dict(GOOD_META)
    meta["projectSettings"] = settings
    return meta


MULTIVIEW_BROKEN = _broken(
    {
        "multiView": {"enabled": True, "tagName": "group", "tagId": None, "isSynced": False},
        "labelingInterface": "bogus",
    }
)
SETTINGS_BROKEN = _broken(
    {
        "multiView": {"enabled": False, "tagName": None, "tagId": None, "isSynced": False},
        "labelingInterface": "bogus",
    }
)


def _run_download(load_main, server_meta, monkeypatch):
    project_api = FakeProjectApi(server_meta, GOOD_META)
    api = make_api(project_api)
    main = load_main(api)
    seen = {}

    def fake_download(api_, project_id, dest_dir, **kwargs):
        # the SDK reads the meta itself with settings; what it gets must deserialize
        seen["meta"] = sly.ProjectMeta.from_json(api_.project.get_meta(project_id, with_settings=True))
        sly.json.dump_json_file(seen["meta"].to_json(), os.path.join(dest_dir, "meta.json"))

    monkeypatch.setattr(main, "download_async_or_sync", fake_download)
    monkeypatch.setattr(main, "enforce_name_limits", lambda d: None)
    project = sly.ProjectInfo(**{f: None for f in sly.ProjectInfo._fields})._replace(
        id=10, name="p", type="images"
    )
    out = main.download(project)
    return project_api, seen, out


def test_broken_multiview_settings_do_not_touch_server_meta(load_main, monkeypatch):
    project_api, seen, out = _run_download(load_main, MULTIVIEW_BROKEN, monkeypatch)
    assert project_api.updated_meta == []
    assert [c.name for c in seen["meta"].obj_classes] == ["car"]
    assert os.path.isfile(os.path.join(out, "meta.json"))


def test_broken_settings_do_not_touch_server_meta(load_main, monkeypatch):
    project_api, seen, _ = _run_download(load_main, SETTINGS_BROKEN, monkeypatch)
    assert project_api.updated_meta == []
    assert [c.name for c in seen["meta"].obj_classes] == ["car"]


def test_valid_meta_is_used_as_is(load_main, monkeypatch):
    project_api, seen, _ = _run_download(load_main, GOOD_META, monkeypatch)
    assert project_api.updated_meta == []
    assert [c.name for c in seen["meta"].obj_classes] == ["car"]


def test_meta_override_is_scoped_to_the_download(load_main, monkeypatch):
    project_api = FakeProjectApi(SETTINGS_BROKEN, GOOD_META)
    api = make_api(project_api)
    main = load_main(api)
    monkeypatch.setattr(main, "download_async_or_sync", lambda *a, **k: None)
    monkeypatch.setattr(main, "enforce_name_limits", lambda d: None)
    project = sly.ProjectInfo(**{f: None for f in sly.ProjectInfo._fields})._replace(
        id=10, name="p", type="images"
    )
    main.download(project)
    # after the export the api object serves the server's meta again
    assert api.project.get_meta(10, with_settings=True) == SETTINGS_BROKEN
