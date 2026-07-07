import asyncio
import os
import re
from collections import defaultdict
from typing import Dict, List

from dotenv import load_dotenv

import sly_functions as f
import supervisely as sly
import workflow as w
from supervisely._utils import generate_free_name
from supervisely.annotation.annotation import AnnotationJsonFields as AJF
from supervisely.api.entities_collection_api import CollectionType, CollectionTypeFilter
from supervisely.annotation.label import LabelJsonFields as LJF
from supervisely.annotation.tag import TagJsonFields as TJF
from supervisely.api.module_api import ApiField
from supervisely.convert.base_converter import AvailableImageConverters
from supervisely.convert.image.overlay.overlay_converter import OverlayImageConverter
from supervisely.io.fs import get_file_ext
from supervisely.project.download import download_async_or_sync
from supervisely.project.project_settings import LabelingInterface

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    sly.logger.setLevel(10)

api = sly.Api.from_env()

# region constants
parent_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(parent_dir, "data")
batch_size = 10
# endregion
# region envvars
team_id = sly.env.team_id()
project_id = sly.env.project_id()
dataset_id = sly.env.dataset_id(raise_not_found=False)
mode = os.environ.get("modal.state.download", "all")
replace_method = (os.environ.get("modal.state.fixExtension", "false")).lower() == "true"
collection_id = os.environ.get("modal.state.collectionId")
preserve_structure = (os.environ.get("modal.state.preserveStructure", "true")).lower() == "true"
flat_dataset_name = os.environ.get("modal.state.datasetName")
# endregion
sly.logger.info(
    f"Team: {team_id}, Project: {project_id}, Dataset: {dataset_id}, Mode: {mode}, "
    f"Fix extension: {replace_method}, Collection: {collection_id}, "
    f"Preserve structure: {preserve_structure}"
)

# A single high-resolution image with a large bitmap/alpha-mask annotation
# can carry a multi-megabyte JSON payload, so batches are capped both by
# item count and by cumulative pixel area rather than item count alone.
COLLECTION_BATCH_MAX_ITEMS = 1000
COLLECTION_BATCH_MAX_PIXELS = 100_000_000  # ~100 MP, e.g. two 8000x6000 images
COLLECTION_BATCH_MIN_ITEMS = 10  # guaranteed batch size even for very large images
APP_NAME = "Export to Supervisely format"
# auto-created filter collections are named like "Filtered entities 2026-07-03T14-31-57-501Z"
FILTERED_COLLECTION_PATTERN = re.compile(
    r"^Filtered entities \d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3}Z"
)


def rename_filtered_collection(collection_info) -> None:
    """Rename an auto-created filter collection to reflect the task that processed it."""
    if not FILTERED_COLLECTION_PATTERN.match(collection_info.name):
        return
    task_id = sly.env.task_id(raise_not_found=False)
    if task_id is None:
        return
    new_name = f"{APP_NAME} (Task {task_id})"
    try:
        api.entities_collection.update(collection_info.id, name=new_name)
        sly.logger.info(
            f"Collection {collection_info.id} renamed: '{collection_info.name}' -> '{new_name}'"
        )
    except Exception as e:
        sly.logger.warning(f"Failed to rename collection {collection_info.id}: {repr(e)}")


def get_collection_image_infos(collection_id: int):
    """Return the collection info and its image infos."""
    collection_info = api.entities_collection.get_info_by_id(collection_id)
    if collection_info is None:
        raise ValueError(f"Collection with id={collection_id} not found")
    collection_type = (
        CollectionTypeFilter.AI_SEARCH
        if collection_info.type == CollectionType.AI_SEARCH
        else CollectionTypeFilter.DEFAULT
    )
    image_infos = api.entities_collection.get_items(
        collection_info.id, collection_type, collection_info.project_id
    )
    if len(image_infos) == 0:
        raise ValueError(f"Collection with id={collection_id} is empty")
    return collection_info, image_infos


def disambiguate_names(image_infos):
    """Rename images whose names repeat across datasets in the flat list.

    Every image involved in a name conflict gets its source dataset ID appended
    to the name, so the origin of each item stays visible.
    """
    progress = sly.Progress(
        "Checking for duplicate names", len(image_infos), need_info_log=True
    )
    name_counts = defaultdict(int)
    for image_info in image_infos:
        name_counts[image_info.name] += 1
    used_names = {image_info.name for image_info in image_infos}
    result = []
    for image_info in image_infos:
        if name_counts[image_info.name] < 2:
            result.append(image_info)
            progress.iter_done_report()
            continue
        if "." in image_info.name:
            stem, ext = image_info.name.rsplit(".", 1)
            new_name = f"{stem}_{image_info.dataset_id}.{ext}"
        else:
            new_name = f"{image_info.name}_{image_info.dataset_id}"
        new_name = generate_free_name(
            used_names, new_name, with_ext=True, extend_used_names=True
        )
        sly.logger.info(
            f"Duplicate image name in collection: '{image_info.name}' "
            f"(dataset {image_info.dataset_id}) renamed to '{new_name}'"
        )
        result.append(image_info._replace(name=new_name))
        progress.iter_done_report()
    return result


def download_images_batch(dataset_id: int, image_ids: List[int], paths: List[str]) -> None:
    """Download images to local paths, preferring the faster async downloader.

    Falls back to the sync download on any error, matching the fallback
    philosophy of download_async_or_sync used elsewhere in this app.
    """
    try:
        coro = api.image.download_paths_async(image_ids, paths)
        loop = sly.utils.get_or_create_event_loop()
        if loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            future.result()
        else:
            loop.run_until_complete(coro)
    except Exception as e:
        sly.logger.warning(
            f"Async image download failed, falling back to sync: {repr(e)}"
        )
        api.image.download_paths(dataset_id, image_ids, paths)


def batched_by_pixels(
    images,
    max_pixels=COLLECTION_BATCH_MAX_PIXELS,
    max_items=COLLECTION_BATCH_MAX_ITEMS,
    min_items=COLLECTION_BATCH_MIN_ITEMS,
):
    """Group images into batches bounded by both item count and total pixel area.

    Annotation size for bitmap/alpha-mask labels scales with image resolution,
    not item count, so a fixed-size item batch can't bound memory on its own
    for large images. min_items guarantees a batch isn't flushed by the pixel
    cap before it reaches that many items (accepting more memory for very
    large images); max_items is a hard ceiling regardless of pixel budget.
    """
    batch = []
    batch_pixels = 0
    for image in images:
        image_pixels = (image.width or 0) * (image.height or 0)
        pixel_cap_hit = len(batch) >= min_items and batch_pixels + image_pixels > max_pixels
        if batch and (pixel_cap_hit or len(batch) >= max_items):
            yield batch
            batch = []
            batch_pixels = 0
        batch.append(image)
        batch_pixels += image_pixels
    if batch:
        yield batch


def download_collection_flat(
    project_meta: sly.ProjectMeta,
    download_dir: str,
    collection_info,
    collection_images,
) -> str:
    """Download collection images into a single dataset in Supervisely format."""
    dataset_name = flat_dataset_name or f"Collection {collection_info.id}"
    sly.json.dump_json_file(project_meta.to_json(), os.path.join(download_dir, "meta.json"))

    dataset_dir = os.path.join(download_dir, dataset_name)
    img_dir = os.path.join(dataset_dir, sly.Dataset.item_dir_name)
    ann_dir = os.path.join(dataset_dir, sly.Dataset.ann_dir_name)
    img_info_dir = os.path.join(dataset_dir, sly.Dataset.item_info_dir_name)
    meta_dir = os.path.join(dataset_dir, sly.Dataset.meta_dir_name)
    sly.fs.mkdir(img_dir)
    sly.fs.mkdir(ann_dir)
    sly.fs.mkdir(img_info_dir)

    collection_images = disambiguate_names(collection_images)
    by_dataset = defaultdict(list)
    for image in collection_images:
        by_dataset[image.dataset_id].append(image)

    progress = sly.Progress("Downloading collection items", len(collection_images))
    for src_dataset_id, images in by_dataset.items():
        for image_batch in batched_by_pixels(images):
            image_ids = [image.id for image in image_batch]
            ann_infos = api.annotation.download_batch(src_dataset_id, image_ids)
            id_to_ann = {ann.image_id: ann.annotation for ann in ann_infos}
            for image in image_batch:
                ann = id_to_ann.get(image.id)
                if ann is None:
                    ann = sly.Annotation((image.height, image.width)).to_json()
                sly.json.dump_json_file(ann, os.path.join(ann_dir, f"{image.name}.json"))
                sly.json.dump_json_file(
                    image._asdict(),
                    os.path.join(img_info_dir, f"{image.name}.json"),
                    indent=4,
                )
                if image.meta:
                    sly.fs.mkdir(meta_dir)
                    sly.json.dump_json_file(
                        image.meta, os.path.join(meta_dir, f"{image.name}.json")
                    )
            if mode == "all":
                paths = [os.path.join(img_dir, image.name) for image in image_batch]
                download_images_batch(src_dataset_id, image_ids, paths)
            progress.iters_done_report(len(image_batch))

    sly.logger.info(
        f"Collection downloaded into a single dataset '{dataset_name}' "
        f"({len(collection_images)} items)"
    )
    return download_dir


def ours_convert_json_info(self, info: dict, skip_missing=True):
    if info is None:
        return None
    temp_ext = None
    field_values = []
    for field_name in self.info_sequence():
        if field_name == ApiField.EXT:
            continue
        if skip_missing is True:
            val = info.get(field_name, None)
        else:
            val = info[field_name]
        field_values.append(val)
        if field_name == ApiField.MIME:
            temp_ext = val.split("/")[1]
            field_values.append(temp_ext)
    for idx, field_name in enumerate(self.info_sequence()):
        if field_name == ApiField.NAME:
            cur_ext = get_file_ext(field_values[idx]).replace(".", "").lower()
            if not cur_ext:
                field_values[idx] = "{}.{}".format(field_values[idx], temp_ext)
                break
            if temp_ext == "jpeg" and cur_ext in ["jpg", "jpeg", "mpo"]:
                break
            if temp_ext != cur_ext and cur_ext is not None:
                pass
            break
    return self.InfoType(*field_values)


if replace_method:
    sly.logger.debug("Fix image extension method is enabled")
    sly.api.image_api.ImageApi._convert_json_info = ours_convert_json_info


def add_additional_label_fields(project_dir: str):
    project = sly.Project(project_dir, sly.OpenMode.READ)
    progress = sly.Progress("Adding additional fields to labels", project.total_items)

    class_names_sanitized = {}
    new_obj_classes = []
    for objclass in project.meta.obj_classes:
        santized_class_name = f.sanitize_name_if_needed(objclass.name)
        if santized_class_name:
            class_names_sanitized[objclass.name] = santized_class_name
            objclass = objclass.clone(name=santized_class_name)
        new_obj_classes.append(objclass)

    tagmeta_names_sanitized = {}
    new_tagmetas = []
    for tagmeta in project.meta.tag_metas:
        santized_tag_name = f.sanitize_name_if_needed(tagmeta.name)
        if santized_tag_name:
            tagmeta_names_sanitized[tagmeta.name] = santized_tag_name
            tagmeta = tagmeta.clone(name=santized_tag_name)
        new_tagmetas.append(tagmeta)

    names_sanitized = len(class_names_sanitized) > 0 or len(tagmeta_names_sanitized) > 0
    if names_sanitized:
        new_meta = project.meta.clone(new_obj_classes, new_tagmetas)
        meta_path = project._get_project_meta_path()
        sly.json.dump_json_file(new_meta.to_json(), meta_path)

    for dataset in project:
        dataset: sly.Dataset

        names = dataset.get_items_names()
        for name in names:
            changed = False
            ann_path = dataset.get_ann_path(name)
            meta_path = dataset.get_item_meta_path(name)
            image_meta = None
            if os.path.exists(meta_path):
                image_meta = sly.json.load_json_file(meta_path)
                try:
                    K_instrinsics = f.get_k_intrinsics_from_meta(image_meta)
                except ValueError:
                    sly.logger.debug("Failed to get K_intrinsics from meta, skipping...")
                    progress.iter_done_report()
                    continue

            # If there is no meta and we haven't sanitized any names, we can skip processing this annotation
            if image_meta is None and names_sanitized is False:
                progress.iter_done_report()
                continue

            ann_json = sly.json.load_json_file(ann_path)
            image_tags = ann_json.get(AJF.IMG_TAGS, [])
            if image_tags and names_sanitized:
                for tag in image_tags:
                    santized_tag_name = tagmeta_names_sanitized.get(tag[ApiField.NAME])
                    if santized_tag_name:
                        tag[ApiField.NAME] = santized_tag_name
                        changed = True

            # todo image tags
            for label in ann_json[AJF.LABELS]:
                if names_sanitized:
                    objclass_name = label[LJF.OBJ_CLASS_NAME]
                    santized_class_name = class_names_sanitized.get(objclass_name)
                    if santized_class_name:
                        label[LJF.OBJ_CLASS_NAME] = santized_class_name
                        changed = True

                    label_tags = label.get(LJF.TAGS)
                    if label_tags:
                        for tag in label_tags:
                            tag_name = tag[TJF.TAG_NAME]
                            s_tag_name = tagmeta_names_sanitized.get(tag_name)
                            if s_tag_name:
                                tag[TJF.TAG_NAME] = s_tag_name
                                changed = True

                if image_meta is not None:
                    if label[LJF.GEOMETRY_TYPE] == sly.Cuboid2d.geometry_name():
                        linestrings = f.get_linestrings_from_label(label, K_instrinsics)
                        label["_curved_cylindrical_edges"] = linestrings
                        changed = True
                    elif label[LJF.GEOMETRY_TYPE] == sly.Polygon.geometry_name():
                        linestrings = f.get_polygon_linestrings(label["points"], K_instrinsics)
                        label["_curved_cylindrical_edges"] = linestrings
                        changed = True

            if changed:
                sly.json.dump_json_file(ann_json, ann_path)
            progress.iter_done_report()


def get_dataset_ids(project: sly.ProjectInfo) -> List[int]:
    if dataset_id is not None:
        dataset_ids = [dataset_id]
        nested_datasets = api.dataset.get_nested(project.id, dataset_id)
        nested_dataset_ids = [dataset.id for dataset in nested_datasets]
        dataset_ids.extend(nested_dataset_ids)
    else:
        datasets = api.dataset.get_list(project.id, recursive=True)
        dataset_ids = [dataset.id for dataset in datasets]
    return dataset_ids


def is_overlay_project(project: sly.ProjectInfo, project_meta: sly.ProjectMeta) -> bool:
    if project.type == AvailableImageConverters.OVERLAY:
        return True

    return project_meta.project_settings.labeling_interface == LabelingInterface.OVERLAY


def make_overlay_dataset_dirs(dataset_dir: str):
    item_dir = os.path.join(dataset_dir, OverlayImageConverter.ITEM_DIR)
    ann_dir = os.path.join(dataset_dir, OverlayImageConverter.ANN_DIR)
    overlays_root = os.path.join(dataset_dir, OverlayImageConverter.OVERLAY_DIR)

    sly.fs.mkdir(item_dir)
    sly.fs.mkdir(ann_dir)
    sly.fs.mkdir(overlays_root)
    return item_dir, ann_dir, overlays_root


def download_overlay_images(
    dataset_id: int,
    images,
    image_paths: Dict[int, str],
    progress_cb=None,
) -> None:
    for image_batch in sly.batched(images, batch_size):
        ids = [image.id for image in image_batch]
        paths = [image_paths[image.id] for image in image_batch]
        for path in paths:
            sly.fs.mkdir(os.path.dirname(path))
        api.image.download_paths(dataset_id, ids, paths, progress_cb=progress_cb)


def download_overlay_annotations(dataset_id: int, parent_images, ann_dir: str) -> None:
    progress = sly.Progress("Downloading overlay annotations", len(parent_images))
    for image_batch in sly.batched(parent_images, batch_size):
        image_ids = [image.id for image in image_batch]
        ann_infos = api.annotation.download_batch(
            dataset_id,
            image_ids,
            progress_cb=progress.iters_done_report,
        )
        id_to_ann = {ann.image_id: ann.annotation for ann in ann_infos}
        for image in image_batch:
            ann = id_to_ann.get(image.id)
            if ann is None:
                ann = sly.Annotation((image.height, image.width)).to_json()
            ann_path = os.path.join(ann_dir, f"{image.name}.json")
            sly.json.dump_json_file(ann, ann_path)


def download_overlay_project(
    project: sly.ProjectInfo,
    project_meta: sly.ProjectMeta,
    download_dir: str,
    dataset_ids: List[int],
    images_ids: List[int] = None,
) -> str:
    sly.json.dump_json_file(project_meta.to_json(), os.path.join(download_dir, "meta.json"))

    if mode != "all":
        sly.logger.warning(
            "Overlay format requires parent and overlay image files. "
            "Exporting images despite the 'only json annotations' option."
        )

    selected_dataset_ids = set(dataset_ids)
    total_images = 0
    for _, dataset in api.dataset.tree(project.id):
        if dataset.id not in selected_dataset_ids:
            continue
        total_images += dataset.images_count

    image_progress = sly.Progress("Downloading overlay images", total_images)
    for parents, dataset in api.dataset.tree(project.id):
        if dataset.id not in selected_dataset_ids:
            continue

        dataset_path_parts = parents + [dataset.name]
        dataset_dir = os.path.join(download_dir, *dataset_path_parts)
        item_dir, ann_dir, overlays_root = make_overlay_dataset_dirs(dataset_dir)

        parent_images = api.image.get_list(
            dataset.id,
            filters=[
                {
                    ApiField.FIELD: ApiField.PARENT_ID,
                    ApiField.OPERATOR: "=",
                    ApiField.VALUE: None,
                }
            ],
            force_metadata_for_links=False,
        )
        overlay_images = api.image.get_list(
            dataset.id,
            filters=[
                {
                    ApiField.FIELD: ApiField.PARENT_ID,
                    ApiField.OPERATOR: "!=",
                    ApiField.VALUE: None,
                }
            ],
            force_metadata_for_links=False,
            extra_fields=[ApiField.PARENT_ID],
        )
        overlay_images_by_parent = defaultdict(list)
        for image in overlay_images:
            overlay_images_by_parent[image.parent_id].append(image)

        if images_ids is not None:
            selected_images_ids = set(images_ids)
            dropped_parents = [
                image for image in parent_images if image.id not in selected_images_ids
            ]
            if dropped_parents:
                skipped_count = len(dropped_parents) + sum(
                    len(overlay_images_by_parent.pop(image.id, []))
                    for image in dropped_parents
                )
                image_progress.iters_done_report(skipped_count)
            parent_images = [
                image for image in parent_images if image.id in selected_images_ids
            ]

        parent_ids = {image.id for image in parent_images}
        orphan_overlay_ids = set(overlay_images_by_parent) - parent_ids
        if orphan_overlay_ids:
            skipped_count = sum(len(overlay_images_by_parent[id]) for id in orphan_overlay_ids)
            sly.logger.warning(
                f"Dataset '{dataset.name}' contains overlays with missing parent images: "
                f"{sorted(orphan_overlay_ids)}. They will be skipped."
            )
            image_progress.iters_done_report(skipped_count)

        image_paths = {}
        for parent in parent_images:
            parent_path = os.path.join(item_dir, parent.name)
            image_paths[parent.id] = parent_path

            parent_name = os.path.splitext(parent.name)[0]
            parent_overlay_dir = os.path.join(overlays_root, parent_name)
            for overlay in overlay_images_by_parent.get(parent.id, []):
                image_paths[overlay.id] = os.path.join(parent_overlay_dir, overlay.name)

        export_images = parent_images + [
            overlay
            for parent in parent_images
            for overlay in overlay_images_by_parent.get(parent.id, [])
        ]
        download_overlay_images(
            dataset.id,
            export_images,
            image_paths,
            progress_cb=image_progress.iters_done_report,
        )
        download_overlay_annotations(dataset.id, parent_images, ann_dir)

        meta_dir = os.path.join(dataset_dir, "meta")
        for image in parent_images:
            if image.meta:
                sly.fs.mkdir(meta_dir)
                sly.json.dump_json_file(image.meta, os.path.join(meta_dir, f"{image.name}.json"))

    sly.logger.info("Overlay project downloaded...")
    return download_dir


def download(project: sly.ProjectInfo) -> str:
    """Downloads the project and returns the path to the downloaded directory.

    :param project: The project to download
    :type project: sly.ProjectInfo
    :return: The path to the downloaded directory
    :rtype: str
    """
    download_dir = os.path.join(data_dir, f"{project.id}_{project.name}")
    sly.fs.mkdir(download_dir, remove_content_if_exists=True)

    images_ids = None
    if collection_id is not None:
        collection_info, collection_images = get_collection_image_infos(int(collection_id))
        rename_filtered_collection(collection_info)
        images_ids = [image.id for image in collection_images]
        dataset_ids = sorted({image.dataset_id for image in collection_images})
        sly.logger.info(
            f"Downloading {len(images_ids)} images of collection {collection_info.id} "
            f"from {len(dataset_ids)} dataset(s)"
        )
    else:
        dataset_ids = get_dataset_ids(project)
    meta_json = api.project.get_meta(project.id, with_settings=True)
    try:
        project_meta = sly.ProjectMeta.from_json(meta_json)
    except Exception:
        f.project_meta_deserialization_check(api, project)
        project_meta = sly.ProjectMeta.from_json(
            api.project.get_meta(project.id, with_settings=True)
        )

    if is_overlay_project(project, project_meta):
        if collection_id is not None and not preserve_structure:
            sly.logger.warning(
                "Flat collection download is not supported for overlay projects. "
                "Dataset structure will be preserved."
            )
        sly.logger.info("Overlay project detected. Starting custom overlay export...")
        return download_overlay_project(
            project, project_meta, download_dir, dataset_ids, images_ids
        )

    if collection_id is not None and not preserve_structure:
        download_collection_flat(project_meta, download_dir, collection_info, collection_images)
        try:
            add_additional_label_fields(download_dir)
        except Exception as e:
            sly.logger.error(f"Error while adding additional fields: {e}")
        return download_dir

    if mode == "all":
        save_images = True
    else:
        save_images = False

    sly.logger.info(f"Starting download of project {project.name} to {download_dir}...")

    download_async_or_sync(
        api,
        project_id,
        dest_dir=download_dir,
        dataset_ids=dataset_ids,
        log_progress=True,
        save_image_meta=True,
        save_images=save_images,
        save_image_info=True,
        images_ids=images_ids,
    )

    try:
        add_additional_label_fields(download_dir)
    except Exception as e:
        sly.logger.error(f"Error while adding additional fields: {e}")

    sly.logger.info("Project downloaded...")
    return download_dir


def main():
    project = api.project.get_info_by_id(project_id)
    download_dir = download(project)
    w.workflow_input(api, project.id)
    file_info = sly.output.set_download(download_dir)
    w.workflow_output(api, file_info)
    sly.logger.info("Archive uploaded and ready for download.")


if __name__ == "__main__":
    sly.main_wrapper("main", main)
