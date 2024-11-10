import asyncio
import os
from distutils import util

import supervisely as sly
from dotenv import load_dotenv
from supervisely.api.module_api import ApiField
from supervisely.io.fs import get_file_ext

import sly_functions as f
import workflow as w

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/supervisely.env"))

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


def add_additional_field_for_cuboid(project_dir: str):
    project = sly.Project(project_dir, sly.OpenMode.READ)
    progress = sly.Progress("Adding additional field for cuboid", project.total_items)
    for dataset in project:
        dataset: sly.Dataset

        names = dataset.get_items_names()
        for name in names:
            changed = False
            ann_path = dataset.get_ann_path(name)
            meta_path = dataset.get_item_meta_path(name)

            ann_json = sly.json.load_json_file(ann_path)
            for label in ann_json["objects"]:
                if label["geometryType"] == sly.Cuboid2d.geometry_name():
                    image_meta = sly.json.load_json_file(meta_path)
                    K_instrinsics = f.get_k_intrinsics_from_meta(image_meta)
                    linestrings = f.get_linestrings_from_label(label, K_instrinsics)
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

    if dataset_id is not None:
        dataset_ids = [dataset_id]
    else:
        datasets = api.dataset.get_list(project.id, recursive=True)
        dataset_ids = [dataset.id for dataset in datasets]

    if mode == "all":
        save_images = True
    else:
        save_images = False

    sly.logger.info(f"Starting download of project {project.name} to {download_dir}...")

    loop = sly.fs.get_or_create_event_loop()
    coro = sly.Project.download_async(
        api,
        project_id,
        dest_dir=download_dir,
        dataset_ids=dataset_ids,
        log_progress=True,
        save_image_meta=True,
        save_images=save_images,
    )
    if loop.is_running():
        sly.logger.debug("Loop is already running, using run_coroutine_threadsafe")
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        future.result()
    else:
        sly.logger.debug("Loop is not running, using run_until_complete")
        loop.run_until_complete(coro)

    meta_path = os.path.join(download_dir, "meta.json")
    meta = sly.ProjectMeta.from_json(sly.json.load_json_file(meta_path))
    if any(obj_cls.geometry_type == sly.Cuboid2d for obj_cls in meta.obj_classes):
        try:
            add_additional_field_for_cuboid(download_dir)
        except Exception as e:
            sly.logger.error(f"Error while adding additional field for 2D cuboid: {e}")

    sly.logger.info("Project downloaded...")
    return download_dir


if __name__ == "__main__":
    project = api.project.get_info_by_id(project_id)
    download_dir = download(project)
    w.workflow_input(api, project.id)
    file_info = sly.output.set_download(download_dir)
    w.workflow_output(api, file_info)
    sly.logger.info("Archive uploaded and ready for download.")
