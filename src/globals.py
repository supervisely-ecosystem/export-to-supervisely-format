import os
import supervisely_lib as sly

api: sly.Api = sly.Api.from_env()
my_app: sly.AppService = sly.AppService()

TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
PROJECT_ID = int(os.environ['modal.state.slyProjectId'])
task_id = int(os.environ["TASK_ID"])
mode = int(os.environ['modal.state.radioButtons'])
replace_method = bool(os.environ['modal.state.checkedButton'])
batch_size = 10
