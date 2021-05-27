import os
import supervisely_lib as sly
import globals as g
from ui import ui

if g.replace_method:
    print('Entered')
    from extension.change_image_api_class_method import ours_convert_json_info
    sly.api.image_api.ImageApi._convert_json_info = ours_convert_json_info


@g.my_app.callback("download_as_sly")
@sly.timeit
def download_as_sly(api: sly.Api, task_id, context, state, app_logger):
    project = g.api.project.get_info_by_id(g.PROJECT_ID)
    datasets = g.api.dataset.get_list(project.id)
    dataset_ids = [dataset.id for dataset in datasets]
    if g.mode == 11:
        download_json_plus_images(api, project, dataset_ids)
    else:
        download_only_json(api, project, dataset_ids)
    download_dir = os.path.join(g.my_app.data_dir, f'{project.id}_{project.name}')
    full_archive_name = str(project.id) + '_' + project.name + '.tar'
    result_archive = os.path.join(g.my_app.data_dir, full_archive_name)
    sly.fs.archive_directory(download_dir, result_archive)
    app_logger.info("Result directory is archived")
    upload_progress = []
    remote_archive_path = "/Download-data/{}_{}".format(task_id, full_archive_name)


    def _print_progress(monitor, upload_progress):
        if len(upload_progress) == 0:
            upload_progress.append(sly.Progress(message="Upload {!r}".format(full_archive_name),
                                                total_cnt=monitor.len,
                                                ext_logger=app_logger,
                                                is_size=True))
        upload_progress[0].set_current_value(monitor.bytes_read)

    file_info = api.file.upload(g.TEAM_ID, result_archive, remote_archive_path,
                                lambda m: _print_progress(m, upload_progress))
    app_logger.info("Uploaded to Team-Files: {!r}".format(file_info.full_storage_url))
    api.task.set_output_archive(task_id, file_info.id, full_archive_name, file_url=file_info.full_storage_url)
    g.my_app.stop()


def download_json_plus_images(api, project, dataset_ids):
    sly.logger.info('DOWNLOAD_PROJECT', extra={'title': project.name})
    download_dir = os.path.join(g.my_app.data_dir, f'{project.id}_{project.name}')
    sly.download_project(api, project.id, download_dir, dataset_ids=dataset_ids,
                         log_progress=True, batch_size=g.batch_size)
    sly.logger.info('Project {!r} has been successfully downloaded.'.format(project.name))


def download_only_json(api, project, dataset_ids):
    sly.logger.info('DOWNLOAD_PROJECT', extra={'title': project.name})
    download_dir = os.path.join(g.my_app.data_dir, f'{project.id}_{project.name}')
    sly.fs.mkdir(download_dir)
    meta_json = api.project.get_meta(project.id)
    sly.io.json.dump_json_file(meta_json, os.path.join(download_dir, 'meta.json'))

    total_images = 0
    dataset_info = (
        [api.dataset.get_info_by_id(ds_id) for ds_id in dataset_ids]
        if (dataset_ids is not None) else api.dataset.get_list(project.id))

    for dataset in dataset_info:
        ann_dir = os.path.join(download_dir, dataset.name, 'ann')
        sly.fs.mkdir(ann_dir)

        images = api.image.get_list(dataset.id)
        ds_progress = sly.Progress(
            'Downloading annotations for: {!r}/{!r}'.format(project.name, dataset.name), total_cnt=len(images))
        for batch in sly.batched(images, batch_size=10):
            image_ids = [image_info.id for image_info in batch]
            image_names = [image_info.name for image_info in batch]

            # download annotations in json format
            ann_infos = api.annotation.download_batch(dataset.id, image_ids)

            for image_name, ann_info in zip(image_names, ann_infos):
                sly.io.json.dump_json_file(ann_info.annotation, os.path.join(ann_dir, image_name + '.json'))
            ds_progress.iters_done_report(len(batch))
            total_images += len(batch)

    sly.logger.info('Project {!r} has been successfully downloaded'.format(project.name))
    sly.logger.info('Total number of images: {!r}'.format(total_images))


def main():
    sly.logger.info(
        "Script arguments",
        extra={
            "TEAM_ID":      g.TEAM_ID,
            "WORKSPACE_ID": g.WORKSPACE_ID,
            "PROJECT_ID":   g.PROJECT_ID
        }
    )
    # image_id = 4949471
    # g.api.image.get_info_by_id(image_id)
    data = {}
    state = {}
    ui.init(data, state)
    g.my_app.run(initial_events=[{"command": "download_as_sly"}])


if __name__ == "__main__":
    sly.main_wrapper("main", main)
