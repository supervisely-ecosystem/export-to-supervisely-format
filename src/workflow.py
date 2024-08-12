# This module contains the functions that are used to configure the input and output of the workflow for the current app.

import supervisely as sly
from typing import Union

def workflow_input(api: sly.Api, project_id: int):
    api.app.workflow.add_input_project(project_id)
    sly.logger.debug(f"Workflow: Input project - {project_id}")

def workflow_output(api: sly.Api, file: Union[int, sly.api.file_api.FileInfo]):
    try:
        if isinstance(file, int):
            file = api.file.get_info_by_id(file)
        meta = {"customRelationSettings": {
                    "icon": {
                        "icon": "zmdi-archive",
                        "color": "#33c94c",
                        "backgroundColor": "#d9f7e4"
                    },
                    "title": f"<h4>{file.name}</h4>",
                    "mainLink": {"url": f"/files/{file.id}/true/?teamId={file.team_id}", "title": "Download"}
                    }}
        api.app.workflow.add_output_file(file, meta=meta)
        sly.logger.debug(f"Workflow: Output file - {file}")
    except Exception as e:
        sly.logger.debug(f"Failed to add output to the workflow: {repr(e)}")