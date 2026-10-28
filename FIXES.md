# Fixes: export-to-supervisely-format (branch `ecosystem-fixes-2026-10`)

Lifecycle `maintained` (46,960 runs): targeted fixes only. Base: `origin/HEAD` 36d8c8d (v3.6.1).
Nothing pushed or released.

## Behaviour contract (unchanged)

Headless, "Download as" on `images_project`, `images_dataset`, `audio_project`, `audio_dataset`.

| input | meaning |
|---|---|
| project / dataset context | whole project, or the dataset and every dataset nested in it |
| `modal.state.download` (`all`) | `all` = images + JSON; `only_json` = annotations, img_info and meta only |
| `modal.state.fixExtension` (`false`) | names without an extension get one from the MIME type |
| `modal.state.entityIds` (JSON list) | only these images (launch from an image selection) |
| `modal.state.collectionId` | images of an entities collection; auto "Filtered entities …" collections renamed to `Export to Supervisely format (Task N)` |
| `modal.state.preserveStructure` (`true`) | selection/collection: `false` puts every image in one dataset |
| `modal.state.datasetName` (`""`) | flat dataset name; empty = `Selected images` / `Collection <id>` |

Output: `<project id>_<project name>.tar` via `sly.output.set_download` (task download link),
SDK project layout: `meta.json`, `<ds>/img|ann|img_info|meta`, nested datasets under
`<ds>/datasets/<child>/…`; overlay projects in the overlay converter layout; audio projects via
`sly.download`. Names over 60 chars are shortened with their sidecars. Class/tag names with
non-printable characters or surrounding whitespace are sanitized in `meta.json` and every
annotation; labels of images with camera calibration in their meta get `_curved_cylindrical_edges`.
Workflow: project in, archive out. **The app never writes to the project.**

## Fixed

| # | commit | bug | test (old → new) |
|---|---|---|---|
| 1 | 5dd448c | **Data loss.** A project meta that failed to deserialize was "repaired" on the server with `api.project.update_meta` (multi-view disabled, or meta replaced by the settings-less one) during a download. Now repaired in memory only and served to the SDK download through `serve_meta`. | `tests/test_meta_readonly.py`: 1/4 → 4/4 |
| 2 | a30f27d | **Broken archive, wrong success.** Name sanitizing skipped every item whose image meta had no calibration, and a curved-edge error on one label (e.g. singular K) aborted the pass after `meta.json` was rewritten; the caller only logged it. Annotations then referenced classes missing from `meta.json` and the archive could not be imported. Now names are sanitized for every item, curved-edge errors skip that label with a warning, and a sanitized name that collides with an existing one is kept as is. | `tests/test_label_fields.py`: 0/3 → 3/3; old/new equivalence on 30 generated items with valid calibration: identical except the items the old code left unsanitized |

Total: `pytest` 7 passed (Python 3.14.3, SDK 6.74.46).

## Intentional changes

- With a broken project meta the export carries the repaired meta (multi-view off, or no
  project settings) and logs that the project was not changed. Before, the project itself was changed.
- Class "car " and class "car" in one project: before, the pass crashed (swallowed) and nothing
  was sanitized; now "car " is kept, other names are still sanitized.

## Found, not fixed

- `only_json` downloads: `enforce_name_limits` walks only `img/` dirs, so annotation/img_info
  sidecars of images with names over 60 chars are not shortened when no images are exported. Only
  matters if the downstream 60-char truncation the code comments describe really exists; not verified.
- `fixExtension=true`: the monkeypatched `ImageApi._convert_json_info` does `mime.split("/")`
  and crashes on an image without a MIME type. Not seen on dev.
- `add_additional_label_fields` is still wrapped in a `try/except` that only logs. After fix 2 the
  remaining failures are I/O-level; turning it into a task failure is a behaviour decision.
- `src/sly_functions.py` imports `cv2` unused (lint only).

## Image

Unchanged: `supervisely/base-py-sdk-hardened:6.74.46`, SDK 6.74.46, `instance_version` 6.18.3.
The code change uses only the standard library and existing SDK calls.

## Dev-check plan

Run each case on the branch and on the published version (latest release, v3.6.1), then compare
the archives (member list + SHA-256 per file; ignore `img_info/*.json` `updated_at`).

| case | command | expected (branch) | published |
|---|---|---|---|
| every shape type | `uv run ecosystem-fleet/fleet.py run export-to-supervisely-format --workspace 733 --project 9390 --version ecosystem-fixes-2026-10 --branch` | `meta.json`, `root/{img,ann,img_info}`, `root/datasets/child/…`, 2 images; identical to published | same command without `--version/--branch` |
| nested, from a parent dataset | `… --project 1317 --dataset 10122 --version ecosystem-fixes-2026-10 --branch` | `Tropical/datasets/{Banana,Mango,Pineapple,Lemon,Orange}`, 16 images; identical to published | same without `--version/--branch` |
| annotations only | `… --project 1317 --state download=only_json --version ecosystem-fixes-2026-10 --branch` | no `img/` files; 22 annotations; identical to published | same without `--version/--branch` |
| flat selection | `… --project 9390 --state 'entityIds=[<both image ids>]' --state preserveStructure=false --state datasetName=flat --version ecosystem-fixes-2026-10 --branch` | one dataset `flat` with 2 images | same without `--version/--branch` |
| sanitized names (setup needed) | copy project 9390 into ws 733 as "agents export-sly sanitize", rename one class to end with a space, set a non-calibration meta (`{"source": "x"}`) on one image, then `… --project <new id> --version ecosystem-fixes-2026-10 --branch` | every `ann/*.json` uses the trimmed class name that `meta.json` has; re-import with main-import succeeds | published: the image with meta keeps the old name; main-import rejects that annotation |

Fix 1 (broken meta) cannot be reproduced on dev without corrupting a project's settings; unit-tested only.
Check in every branch run that the source project's `updatedAt` and meta are unchanged afterwards.
