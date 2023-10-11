import os
import tarfile
from tqdm import tqdm
import supervisely as sly
from supervisely.api.module_api import ApiField
from supervisely.io.fs import get_file_ext
from supervisely.app.v1.app_service import AppService
from distutils import util
from dotenv import load_dotenv

from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000

from dataset_tools import ProjectRepo

if sly.is_development():
    load_dotenv("local.env")
    load_dotenv(os.path.expanduser("~/ninja.env"))


api: sly.Api = sly.Api.from_env()
my_app: AppService = AppService()


TEAM_ID = int(os.environ["context.teamId"])
WORKSPACE_ID = int(os.environ["context.workspaceId"])
PROJECT_ID = int(os.environ["modal.state.slyProjectId"])
DATASET_ID = os.environ.get("modal.state.slyDatasetId", None)
if DATASET_ID is not None:
    DATASET_ID = int(DATASET_ID)

task_id = int(os.environ["TASK_ID"])
mode = os.environ["modal.state.download"]
replace_method = bool(util.strtobool(os.environ["modal.state.fixExtension"]))
batch_size = 10


def pack_directory_to_tar(source_dir, output_tar):
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory '{source_dir}' does not exist.")

    with tarfile.open(output_tar, "w") as tar:
        for root, _, files in os.walk(source_dir):
            with tqdm.tqdm(
                desc=f"Unpacking '{output_tar}'",
                total=len(files),
                unit="file",
            ) as pbar:
                for file in files:
                    file_path = os.path.join(root, file)
                    tar.add(file_path, arcname=os.path.relpath(file_path, source_dir))
                    pbar.update(1)


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


def init(data, state):
    state["download"] = mode
    state["fixExtension"] = replace_method


if replace_method:
    sly.logger.debug("change SDK method")
    sly.api.image_api.ImageApi._convert_json_info = ours_convert_json_info


@my_app.callback("download_as_sly")
@sly.timeit
def download_as_sly(api: sly.Api, task_id, context, state, app_logger):
    global TEAM_ID, PROJECT_ID, DATASET_ID, mode
    project = api.project.get_info_by_id(PROJECT_ID)
    if project is None:
        raise Exception(f"Project with ID {PROJECT_ID} not found in your account")
    if DATASET_ID is not None:
        dataset_ids = [DATASET_ID]
        dataset = api.dataset.get_info_by_id(DATASET_ID)
        if dataset is None:
            raise Exception(f"Dataset with ID {DATASET_ID} not found in your account")
    else:
        try:
            datasets = api.dataset.get_list(project.id)
        except Exception as e:
            raise Exception(
                f"Failed to get list of datasets from project ID:{project.id}. {e}"
            )
        dataset_ids = [dataset.id for dataset in datasets]
    if mode == "all":
        download_json_plus_images(api, project, dataset_ids)
    else:
        download_only_json(api, project, dataset_ids)

    download_dir = os.path.join(my_app.data_dir, f"{project.id}_{project.name}")
    full_archive_name = str(project.id) + "_" + project.name + ".tar"
    result_archive = os.path.join(my_app.data_dir, full_archive_name)
    pack_directory_to_tar(download_dir, result_archive)
    # sly.fs.archive_directory(download_dir, result_archive)
    app_logger.info("Result directory is archived")
    upload_progress = []
    remote_archive_path = os.path.join(
        sly.team_files.RECOMMENDED_EXPORT_PATH,
        "export-to-supervisely-format/{}_{}".format(task_id, full_archive_name),
    )

    def _print_progress(monitor, upload_progress):
        if len(upload_progress) == 0:
            upload_progress.append(
                sly.Progress(
                    message="Upload {!r}".format(full_archive_name),
                    total_cnt=monitor.len,
                    ext_logger=app_logger,
                    is_size=True,
                )
            )
        upload_progress[0].set_current_value(monitor.bytes_read)

    file_info = api.file.upload(
        TEAM_ID,
        result_archive,
        remote_archive_path,
        lambda m: _print_progress(m, upload_progress),
    )
    app_logger.info("Uploaded to Team-Files: {!r}".format(file_info.path))
    api.task.set_output_archive(
        task_id, file_info.id, full_archive_name, file_url=file_info.storage_path
    )
    my_app.stop()


def download_json_plus_images(api, project, dataset_ids):
    sly.logger.info("DOWNLOAD_PROJECT", extra={"title": project.name})
    download_dir = os.path.join(my_app.data_dir, f"{project.id}_{project.name}")
    if os.path.exists(download_dir):
        sly.fs.clean_dir(download_dir)
    sly.download_project(
        api,
        project.id,
        download_dir,
        dataset_ids=dataset_ids,
        log_progress=True,
        batch_size=batch_size,
    )
    sly.logger.info(
        "Project {!r} has been successfully downloaded.".format(project.name)
    )

    sly.logger.info("Start building files...")
    print(project.custom_data)
    build_license(project.custom_data["LICENSE"], download_dir)
    build_readme(project.custom_data["README"], download_dir)
    sly.logger.info("'LICENSE.md' and 'README.md' were successfully built.")


def download_only_json(api, project, dataset_ids):
    sly.logger.info("DOWNLOAD_PROJECT", extra={"title": project.name})
    download_dir = os.path.join(my_app.data_dir, f"{project.id}_{project.name}")
    sly.fs.mkdir(download_dir)
    meta_json = api.project.get_meta(project.id)
    sly.io.json.dump_json_file(meta_json, os.path.join(download_dir, "meta.json"))

    total_images = 0
    dataset_info = (
        [api.dataset.get_info_by_id(ds_id) for ds_id in dataset_ids]
        if (dataset_ids is not None)
        else api.dataset.get_list(project.id)
    )

    for dataset in dataset_info:
        ann_dir = os.path.join(download_dir, dataset.name, "ann")
        sly.fs.mkdir(ann_dir)

        images = api.image.get_list(dataset.id)
        ds_progress = sly.Progress(
            "Downloading annotations for: {!r}/{!r}".format(project.name, dataset.name),
            total_cnt=len(images),
        )
        for batch in sly.batched(images, batch_size=10):
            image_ids = [image_info.id for image_info in batch]
            image_names = [image_info.name for image_info in batch]

            # download annotations in json format
            ann_infos = api.annotation.download_batch(dataset.id, image_ids)

            for image_name, ann_info in zip(image_names, ann_infos):
                sly.io.json.dump_json_file(
                    ann_info.annotation, os.path.join(ann_dir, image_name + ".json")
                )
            ds_progress.iters_done_report(len(batch))
            total_images += len(batch)

    sly.logger.info(
        "Project {!r} has been successfully downloaded".format(project.name)
    )
    sly.logger.info("Total number of images: {!r}".format(total_images))


def build_license(license_content: str, download_dir: str):
    license_path = os.path.join(download_dir, "LICENSE.md")
    print(license_content)
    with open(license_path, "w") as license_file:
        license_file.write(license_content)


def build_readme(readme_content, download_dir):
    readme_path = os.path.join(download_dir, "README.md")
    with open(readme_path, "w") as license_file:
        license_file.write(readme_content)


def main():
    sly.logger.info(
        "Script arguments",
        extra={
            "TEAM_ID": TEAM_ID,
            "WORKSPACE_ID": WORKSPACE_ID,
            "PROJECT_ID": PROJECT_ID,
        },
    )

    data = {}
    state = {}
    init(data, state)
    my_app.run(initial_events=[{"command": "download_as_sly"}])


if __name__ == "__main__":
    sly.main_wrapper("main", main)
