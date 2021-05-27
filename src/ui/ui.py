import supervisely_lib as sly
import globals as g


def init(data, state):
    data['radioButtons'] = g.mode
    state['radioButtons'] = g.mode
    state['checked'] = g.replace_method
