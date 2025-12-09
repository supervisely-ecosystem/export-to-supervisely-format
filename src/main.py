import os
from distutils import util

import supervisely as sly
from dotenv import load_dotenv
from supervisely.annotation.annotation import AnnotationJsonFields as AJF
from supervisely.annotation.label import LabelJsonFields as LJF
from supervisely.api.module_api import ApiField
from supervisely.io.fs import get_file_ext
from supervisely.project.download import download_async_or_sync

import sly_functions as f
import workflow as w

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
replace_method = bool(util.strtobool(os.environ.get("modal.state.fixExtension", "false")))
# endregion
sly.logger.info(
    f"Team: {team_id}, Project: {project_id}, Dataset: {dataset_id}, Mode: {mode}, "
    f"Fix extension: {replace_method}"
)


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
    for objclass in project.meta.obj_classes.items():
        if sanitized_class_name := f.sanitize_name_if_needed(objclass.name):
            class_names_sanitized[objclass.name] = sanitized_class_name
            objclass = objclass.clone(name=sanitized_class_name)
        new_obj_classes.append(objclass)

    tagmeta_names_sanitized = {}
    new_tagmetas = []
    for tagmeta in project.meta.tag_metas.items():
        if sanitized_tag_name := f.sanitize_name_if_needed(tagmeta.name):
            tagmeta_names_sanitized[tagmeta.name] = sanitized_tag_name
            tagmeta = tagmeta.clone(name=sanitized_tag_name)
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

            if image_meta is None and names_sanitized is False:
                progress.iter_done_report()
                continue

            ann_json = sly.json.load_json_file(ann_path)
            image_tags = ann_json.get(AJF.IMG_TAGS, [])
            if image_tags and names_sanitized:
                for tag in image_tags:
                    if santized_tag_name := tagmeta_names_sanitized.get(tag[ApiField.NAME]):
                        tag[ApiField.NAME] = santized_tag_name
                        changed = True
            # todo image tags
            for label in ann_json[AJF.LABELS]:
                if names_sanitized:
                    objclass_name = label[LJF.OBJ_CLASS_NAME]
                    if santized_class_name := class_names_sanitized.get(objclass_name):
                        label[LJF.OBJ_CLASS_NAME] = santized_class_name
                        changed = True

                    if label_tags := label.get(LJF.TAGS):
                        for tag in label_tags:
                            tag_name = tag[LJF.TAG_NAME]
                            if s_tag_name := tagmeta_names_sanitized.get(tag_name):
                                tag[LJF.TAG_NAME] = s_tag_name
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


def download(project: sly.Project) -> str:
    """Downloads the project and returns the path to the downloaded directory.

    :param project: The project to download
    :type project: sly.Project
    :return: The path to the downloaded directory
    :rtype: str
    """
    download_dir = os.path.join(data_dir, f"{project.id}_{project.name}")
    sly.fs.mkdir(download_dir, remove_content_if_exists=True)

    f.project_meta_deserialization_check(api, project)

    if dataset_id is not None:
        dataset_ids = [dataset_id]
        nested_datasets = api.dataset.get_nested(project.id, dataset_id)
        nested_dataset_ids = [dataset.id for dataset in nested_datasets]
        dataset_ids.extend(nested_dataset_ids)
    else:
        datasets = api.dataset.get_list(project.id, recursive=True)
        dataset_ids = [dataset.id for dataset in datasets]

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
