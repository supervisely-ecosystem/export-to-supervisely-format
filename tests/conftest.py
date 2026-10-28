"""Offline test harness: imports src/main.py against a small in-memory fake of sly.Api.

No Supervisely server is contacted. Run from the repo root with the SDK from
dev_requirements.txt installed (plus pytest):
    python -m pytest tests
"""

import copy
import importlib
import os
import sys
from types import SimpleNamespace

import pytest
import supervisely as sly

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
APP_MODULES = ("main", "sly_functions", "workflow")


class FakeProjectApi:
    """Serves one project's meta; records every write the app attempts."""

    def __init__(self, meta_with_settings: dict, meta_without_settings: dict):
        self.meta_with_settings = meta_with_settings
        self.meta_without_settings = meta_without_settings
        self.updated_meta = []

    def get_meta(self, id, with_settings=False):
        src = self.meta_with_settings if with_settings else self.meta_without_settings
        return copy.deepcopy(src)

    def update_meta(self, id, meta):
        # like the server: the next get_meta returns what was written
        self.updated_meta.append((id, meta))
        self.meta_with_settings = meta.to_json()


def make_api(project_api=None):
    return SimpleNamespace(
        project=project_api,
        dataset=SimpleNamespace(get_list=lambda project_id, recursive=False: []),
    )


@pytest.fixture
def load_main(monkeypatch, tmp_path):
    """Return a loader: load_main(api, **env) -> freshly imported `main` module."""

    def _load(api, **env):
        base = {
            "ENV": "production",  # keeps main.py from loading local.env / ~/supervisely.env
            "TEAM_ID": "1",
            "WORKSPACE_ID": "2",
            "PROJECT_ID": "10",
        }
        base.update({k: str(v) for k, v in env.items()})
        for key in list(os.environ):
            if key.startswith("modal.state.") or key in ("DATASET_ID", "context.datasetId"):
                monkeypatch.delenv(key, raising=False)
        for key, value in base.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setattr(sly.Api, "from_env", classmethod(lambda cls, *a, **k: api))
        monkeypatch.syspath_prepend(SRC_DIR)
        for name in APP_MODULES:
            sys.modules.pop(name, None)
        main = importlib.import_module("main")
        monkeypatch.setattr(main, "data_dir", str(tmp_path / "data"))
        return main

    yield _load
    for name in APP_MODULES:
        sys.modules.pop(name, None)
