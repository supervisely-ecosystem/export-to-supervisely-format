import os
from distutils import util

import supervisely as sly
from dotenv import load_dotenv
from supervisely.api.module_api import ApiField
from supervisely.io.fs import get_file_ext

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

    sly.Project.download(
        api,
        project_id,
        dest_dir=download_dir,
        dataset_ids=dataset_ids,
        log_progress=True,
        batch_size=batch_size,
        save_image_meta=True,
        save_images=save_images,
    )

    sly.logger.info("Project downloaded...")
    return download_dir


if __name__ == "__main__":
    project = api.project.get_info_by_id(project_id)
    download_dir = download(project)
    sly.output.set_download(download_dir)
    sly.logger.info("Archive uploaded and ready for download.")
