import os
import time

import supervisely as sly


def _download_batch_with_retry(api: sly.Api, dataset_id, image_ids):
    retry_cnt = 5
    curr_retry = 1
    try:
        imgs_bytes = api.image.download_bytes(dataset_id, image_ids)
        if len(imgs_bytes) != len(image_ids):
            raise RuntimeError(
                f"Downloaded {len(imgs_bytes)} images, but {len(image_ids)} expected."
            )
        return imgs_bytes
    except Exception as e:
        sly.logger.warn(f"Failed to download images... Error: {e}")
        while curr_retry <= retry_cnt:
            try:
                sly.logger.warn(f"Retry {curr_retry}/{retry_cnt} to download images")
                time.sleep(2 * curr_retry)
                imgs_bytes = api.image.download_bytes(dataset_id, image_ids)
                if len(imgs_bytes) != len(image_ids):
                    raise RuntimeError(
                        f"Downloaded {len(imgs_bytes)} images, but {len(image_ids)} expected."
                    )
                return imgs_bytes
            except Exception as e:
                curr_retry += 1
    raise RuntimeError(
        f"Failed to download images with ids {image_ids}. Check your data and try again later."
    )


def download_project(
    api: sly.Api,
    project_id,
    dest_dir,
    dataset_ids=None,
    log_progress=True,
    batch_size=10,
    save_image_meta=True,
):
    dataset_ids = set(dataset_ids) if (dataset_ids is not None) else None
    project_fs = sly.Project(dest_dir, sly.OpenMode.CREATE)
    meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    project_fs.set_meta(meta)

    for dataset_info in api.dataset.get_list(project_id):
        dataset_id = dataset_info.id
        if dataset_ids is not None and dataset_id not in dataset_ids:
            continue

        dataset_fs = project_fs.create_dataset(dataset_info.name)
        images = api.image.get_list(dataset_id)

        if save_image_meta:
            meta_dir = os.path.join(dest_dir, dataset_info.name, "meta")
            sly.fs.mkdir(meta_dir)
            for image_info in images:
                meta_paths = os.path.join(meta_dir, image_info.name + ".json")
                sly.json.dump_json_file(image_info.meta, meta_paths)

        ds_progress = None
        if log_progress:
            ds_progress = sly.Progress(
                "Downloading dataset: {!r}".format(dataset_info.name),
                total_cnt=len(images),
            )

        for batch in sly.batched(images, batch_size=batch_size):
            image_ids = [image_info.id for image_info in batch]
            image_names = [image_info.name for image_info in batch]

            # download images
            batch_imgs_bytes = _download_batch_with_retry(api, dataset_id, image_ids)

            # download annotations in json format
            ann_infos = api.annotation.download_batch(dataset_id, image_ids)
            ann_jsons = [ann_info.annotation for ann_info in ann_infos]

            for name, img_bytes, ann in zip(image_names, batch_imgs_bytes, ann_jsons):
                dataset_fs.add_item_raw_bytes(item_name=name, item_raw_bytes=img_bytes, ann=ann)

            if log_progress:
                ds_progress.iters_done_report(len(batch))
