# import json
from typing import List, Optional, Tuple

import cv2
import numpy as np
import numpy.linalg as la
import supervisely as sly
import math

# from PIL import Image, ImageDraw

# import math
# from pyquaternion import Quaternion
# from scipy.spatial.transform import Rotation as SciRot


BACKPROJ_PRECISION = 1e-4

CUBOID_POINTS = [
    [-1, -1, -1],  # rear, right, bottom
    [-1, -1, 1],  # rear, right, top
    [-1, 1, -1],  # rear, left, bottom
    [-1, 1, 1],  # rear, left, top
    [1, -1, -1],  # front, right, bottom
    [1, -1, 1],  # front, right, top
    [1, 1, -1],  # front, left, bottom
    [1, 1, 1],  # front, left, top
]

# cuboid edges
CUBOID_LINE_INDICES = {
    "face2-bottomright-face2-topright": (0, 1),
    "face2-bottomright-face2-bottomleft": (0, 2),
    "face2-bottomright-face1-bottomright": (0, 4),
    "face2-topright-face2-topleft": (1, 3),
    "face2-topright-face1-topright": (1, 5),
    "face2-bottomleft-face2-topleft": (2, 3),
    "face2-bottomleft-face1-bottomleft": (2, 6),
    "face2-topleft-face1-topleft": (3, 7),
    "face1-bottomright-face1-topright": (4, 5),
    "face1-bottomright-face1-bottomleft": (4, 6),
    "face1-topright-face1-topleft": (5, 7),
    "face1-bottomleft-face1-topleft": (6, 7),
    # (0, 3),  # X at the back
    # (1, 2),  # X at the back
}


# class PILImageDraw:
#     """Context manager which provides a PIL image draw
#     and at exit it updates the numpy array image from which it was created
#     """

#     def __init__(self, img: np.ndarray):
#         self.img = img
#         self.img_pil = Image.fromarray(img)
#         self.img_draw = ImageDraw.Draw(self.img_pil)

#     def __enter__(self):
#         return self.img_draw

#     def __exit__(self, type, value, traceback):
#         np.copyto(self.img, np.asarray(self.img_pil))


def to_hom_coords(pts):
    """converts the coordinates to homogeneous (appends ones)"""

    pts_hom = np.concatenate([pts, np.ones(pts.shape[:-1] + (1,), dtype=pts.dtype)], axis=-1)

    return pts_hom


def from_hom_coords(pts_hom):
    eps = 1e-9

    pts = pts_hom[..., :-1] / (pts_hom[..., -1:] + eps)

    return pts


def backproject_to_ray(img_pts, Ks):
    assert img_pts.shape[-1] == 2
    assert Ks.shape[-2:] == (3, 3)

    iKs = np.linalg.inv(Ks)

    norm_img_pts = (iKs @ to_hom_coords(img_pts)[..., np.newaxis])[..., 0]

    cam_x = np.sin(norm_img_pts[..., [0]])
    cam_y = norm_img_pts[..., [1]]
    cam_z = np.cos(norm_img_pts[..., [0]])

    return np.concatenate([cam_x, cam_y, cam_z], axis=-1)


def project_3d_to_2d(XYZ_pts, K):
    """Projects 3D points into 2D image points with cylindric projection in a
    functional programming style (no state)
    Call this inside pytorch networks where all parameters come from outside (the read batch).
    Broadcasting can be used here.
    Args:
        XYZ_pts (TensorType): [..., 3] or [..., 4] The input (hom) 3D points in camera frame.
        K (TensorType): [..., 3, 3] The camera matrix. To broadcast it make it [L, 1, 3, 3]
        for XYZ_pts[N, 3] points to get [L, N, 2] 2d points, or [N, 3, 3] to get [N, 2] points
        with different K matrices for each point.

    Returns:
        TensorType: [..., 2]. 2D image points. Broadcasting of tensors apply.

    Usage example:
    >>> XYZ_pts = np.array([[0.0, 0.0, 1.0]])
    >>> K = np.eye(3)
    >>> CylindricCameraIntrinsics.project_func(XYZ_pts, K)
    array([[0., 0.]])
    """

    """pure functional version, call this inside pytorch networks"""
    assert XYZ_pts.shape[-1] in (
        3,
        4,
    )
    assert K.shape[-2:] == (3, 3)

    length_xz = np.sqrt(np.sum(XYZ_pts[..., [0, 2]] ** 2, axis=-1, keepdims=True))

    # theta = angle in 2D between camera axis (Z) and ...
    # projection of line from camera center to (xw, yw, zw) on XZ plane [rad]
    theta = np.arctan2(XYZ_pts[..., [0]], XYZ_pts[..., [2]])

    uim_y = XYZ_pts[..., [1]] / length_xz

    img_pts_norm_hom = to_hom_coords(np.concatenate([theta, uim_y], axis=-1))

    # do batched matrix multiplication for which broadcasting can work
    return from_hom_coords((K @ img_pts_norm_hom[..., np.newaxis])[..., 0])


def interpolate_linesegs_on_sphere(
    pts1: np.ndarray, pts2: np.ndarray, angle_res_deg: float = 5.0, offset: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolates line segments in a cam frame into a LineString (sequece of line segments)
        so that no line segment covers more than `angle_res_deg`.
        This equals to interpolation on the surface of a sphere.
        This functions work in a batched mode.
        This function is required to properly display line segments on non-perspective cameras.

    Args:
        pts1 (np.ndarray): [N, 3] start points of line segments
        pts2 (np.ndarray): [N, 3] end points of line segments
        angle_res_deg (float, optional): The angular resolution in degrees. Defaults to 5.0.
        offset (float, optional): The angle (given in radians) with which the bended line should reach beyond the
            2 endpoints. Defaults to 0.0.

    Returns:
        (np.ndarray, np.ndarray): [N, L, 3] A linestring for each linesegment and a mask indicating where it ends.
    """

    assert pts1.shape[-1] == 3
    assert len(pts1.shape) == 2
    assert pts1.shape == pts2.shape
    EPS = 1e-9

    n_planes = np.cross(pts1, pts2)  # the normal vector of planes
    plane_abs = la.norm(n_planes, axis=-1, keepdims=True)

    # add a tiny offset in case the two points are identical
    n_planes = np.where(plane_abs < EPS, EPS / np.sqrt(3), n_planes)
    n_planes /= la.norm(n_planes, axis=-1, keepdims=True)

    # base1 is the pts1
    base_1 = pts1 / la.norm(pts1, axis=-1, keepdims=True)
    base_2 = np.cross(n_planes, pts1)  # the 2nd base vector on the plane
    base_2 /= la.norm(base_2, axis=-1, keepdims=True)

    # the angles on the plane
    alphas = np.arctan2(
        np.sum(base_2 * pts2, axis=-1, keepdims=True),
        np.sum(base_1 * pts2, axis=-1, keepdims=True),
    )
    max_alpha = alphas.max()

    angle_res_rad = np.deg2rad(angle_res_deg)

    eps = 1e-9

    # extend dimensions to [N, L, 1]
    alpha_steps = np.arange(
        0.0 - offset, max_alpha + offset + angle_res_rad - eps, angle_res_rad
    ).reshape([1, -1, 1])
    alphas_b = alphas.reshape(-1, 1, 1).repeat(alpha_steps.shape[1], 1)

    point_mask = alpha_steps < alphas_b + offset + angle_res_rad - eps

    # so we don't go outside
    alpha_steps_trunc = np.minimum(alpha_steps, alphas_b + offset)

    sin_steps = np.sin(alpha_steps_trunc)
    cos_steps = np.cos(alpha_steps_trunc)

    base_1_b = np.expand_dims(base_1, axis=-2)
    base_2_b = np.expand_dims(base_2, axis=-2)

    linesegs = cos_steps * base_1_b + sin_steps * base_2_b

    return linesegs, np.broadcast_to(point_mask, linesegs.shape)


def object_img_linestrings(
    points: np.ndarray,
    indices: List[Tuple[int, int]],
    cam_intrinsics=None,
    dtheta_deg: float = 5.0,
    offset: float = 0.0,
) -> List[np.ndarray]:
    """Computes the coordinates of linestring

    Args:
        points (np.ndarray): the 2D pixel coordinates
        indices: list of tuples with indices (like LINE_INDICES)
        cam (Optional[CameraIntrinsics], optional): The camera intrinsic.
            Defaults to None, in this case points are connected with straight lines
        dtheta_deg (float, optional): the maximum angle for each linesegment.
            Defaults to 5.0.
        offset (float, optional): The angle with which the bended line should reach
            beyond the 2 endpoints. Defaults to 0.0.

    Returns:
        List[np.ndarray]: List of shape[N, 2] linestrings
    """
    NUM_PTS = points.shape[0]
    assert points.shape == (
        NUM_PTS,
        2,
    ), f"points shape should be {(NUM_PTS, 2)}, but it's {points.shape}"

    idx = np.array(list(indices.values()))
    start_pts = points[idx[:, 0]]
    end_pts = points[idx[:, 1]]

    # bended lines equally divided along a 3D sphere
    start_pts_cam = backproject_to_ray(start_pts, cam_intrinsics)
    end_pts_cam = backproject_to_ray(end_pts, cam_intrinsics)
    line_strings_cam, masks = interpolate_linesegs_on_sphere(
        start_pts_cam, end_pts_cam, angle_res_deg=dtheta_deg, offset=offset
    )

    linestrings = [
        project_3d_to_2d(ls_cam[masks[lidx]].reshape(-1, 3), cam_intrinsics)
        for lidx, ls_cam in enumerate(line_strings_cam)
    ]
    res = {edge: linestring.tolist() for edge, linestring in zip(indices.keys(), linestrings)}
    return res


# img_name = "Left_cyl.png"
# img_name = 'Rear_cyl.png'

# # cylindrical_image = cv2.imread(f'/data/cylindrical/{img_name}')
# cylindrical_image = cv2.imread(f"/data/cylindrical/{img_name}")[:, :, ::-1]

# with open(f"/data/cylindrical/{img_name}.json") as f:
#     calib = json.load(f)

# with open(f"/data/cylindrical/labels_{img_name}.json") as f:
#     labels = json.load(f)["annotation"]["objects"]


def get_k_intrinsics_from_meta(meta):
    if "calibration" not in meta:
        raise ValueError("Not found 'calibration' field in image meta")
    calibration = meta["calibration"]
    if "intrinsic" not in calibration:
        raise ValueError("Not found 'intrinsic' field in calibration")
    instrinsics = calibration["intrinsic"]
    fx = instrinsics.get("fx", instrinsics.get("focalLengthX"))
    fy = instrinsics.get("fy", instrinsics.get("focalLengthY"))
    cx = instrinsics.get("cx", instrinsics.get("prinAxisX"))
    cy = instrinsics.get("cy", instrinsics.get("prinAxisY"))
    if any(val is None for val in (fx, fy, cx, cy)):
        raise ValueError(f"Missing values in instrinsics: {instrinsics}")
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def get_linestrings_from_label(label, K_intrinsics):
    points_dict = label["vertices"]
    points = np.asarray(
        [
            [
                points_dict["face2-bottomright"]["loc"][0],
                points_dict["face2-bottomright"]["loc"][1],
            ],
            [
                points_dict["face2-topright"]["loc"][0],
                points_dict["face2-topright"]["loc"][1],
            ],
            [
                points_dict["face2-bottomleft"]["loc"][0],
                points_dict["face2-bottomleft"]["loc"][1],
            ],
            [
                points_dict["face2-topleft"]["loc"][0],
                points_dict["face2-topleft"]["loc"][1],
            ],
            [
                points_dict["face1-bottomright"]["loc"][0],
                points_dict["face1-bottomright"]["loc"][1],
            ],
            [
                points_dict["face1-topright"]["loc"][0],
                points_dict["face1-topright"]["loc"][1],
            ],
            [
                points_dict["face1-bottomleft"]["loc"][0],
                points_dict["face1-bottomleft"]["loc"][1],
            ],
            [
                points_dict["face1-topleft"]["loc"][0],
                points_dict["face1-topleft"]["loc"][1],
            ],
        ]
    )

    linestrings = object_img_linestrings(points, CUBOID_LINE_INDICES, K_intrinsics)
    return linestrings


# with PILImageDraw(cylindrical_image) as img_draw:
#     for label in labels:
#         linestrings = get_linestrings_from_label(label)
#         for linestring in linestrings:
#             img_draw.line(linestring.flatten().tolist(), fill=(255, 0, 0), width=3)

# cv2.imwrite(
#     f"/data/cylindrical/{img_name}_drawn.png", cv2.cvtColor(cylindrical_image, cv2.COLOR_RGB2BGR)
# )


def project_meta_deserialization_check(api: sly.Api, project: sly.ProjectInfo) -> None:
    from supervisely.project.project_meta import ProjectMetaJsonFields
    from supervisely.project.project_settings import ProjectSettingsJsonFields

    meta_json = api.project.get_meta(project.id, with_settings=True)
    try:
        meta = sly.ProjectMeta.from_json(meta_json)
    except Exception as e:
        sly.logger.warning(f"Error while getting project meta: {repr(e)}")
        sly.logger.info("Trying to fix project meta...")

        settings = meta_json.get(ProjectMetaJsonFields.PROJECT_SETTINGS)
        if settings is not None:
            multi_view = settings.get(ProjectSettingsJsonFields.MULTI_VIEW)
            if multi_view.get(ProjectSettingsJsonFields.ENABLED) is True:
                # try to fix the project settings by disabling the multi-view
                multi_view[ProjectSettingsJsonFields.ENABLED] = False
                try:
                    meta = sly.ProjectMeta.from_json(meta_json)
                    api.project.update_meta(project.id, meta)
                    return
                except Exception as e:
                    pass

        # try to fix the project meta by removing the project settings
        sly.logger.info("Trying to get project meta without settings...")
        meta_json = api.project.get_meta(project.id)
        meta = sly.ProjectMeta.from_json(meta_json)
        api.project.update_meta(project.id, meta)

class CylindricalProjection:
    def __init__(self, meta):
        self.intrinsic = meta["calibration"]["intrinsic"]

    def project2d_to_3d_cyl(self, pos):
        x, y = pos['x'], pos['y']
        theta = (x - self.intrinsic['cx']) / self.intrinsic['fx']
        h = (y - self.intrinsic['cy']) / self.intrinsic['fy']
        return np.array([math.cos(theta), h, math.sin(theta)])

    def project3d_to_2d_cyl(self, point3d):
        theta = math.atan2(point3d[2], point3d[0])
        h = point3d[1]
        x = theta * self.intrinsic['fx'] + self.intrinsic['cx']
        y = h * self.intrinsic['fy'] + self.intrinsic['cy']
        return (x, y)

    @staticmethod
    def interpolate_line_segments_on_cylinder(pt1, pt2, segments=10):
        theta1 = math.atan2(pt1[2], pt1[0])
        h1 = pt1[1]
        theta2 = math.atan2(pt2[2], pt2[0])
        h2 = pt2[1]

        dtheta = theta2 - theta1
        if dtheta > math.pi:
            dtheta -= 2 * math.pi
        elif dtheta < -math.pi:
            dtheta += 2 * math.pi

        interpolated_points = []
        for i in range(segments):
            t = i / (segments - 1)
            theta = theta1 + t * dtheta
            h = h1 + t * (h2 - h1)
            interpolated_points.append(np.array([math.cos(theta), h, math.sin(theta)]))
        return interpolated_points

    def interpolate_points_cylindrical(self, points_arr):
        import uuid

        result = []
        n_points = len(points_arr)
        for p_idx, cur_p in enumerate(points_arr):
            next_p = points_arr[(p_idx + 1) % n_points]

            # Project the current point and the next point from 2D to 3D
            pt3d_1 = self.project2d_to_3d_cyl(cur_p['position'])
            pt3d_2 = self.project2d_to_3d_cyl(next_p['position'])

            # Interpolate along the cylinder from pt3d_1 to pt3d_2
            line_segments = self.interpolate_line_segments_on_cylinder(pt3d_1, pt3d_2, segments=10)

            for pt3d in line_segments:
                # Project the interpolated 3D point back to 2D
                pt2d = self.project3d_to_2d_cyl(pt3d)
                result.append({
                    'id': str(uuid.uuid4()),
                    'position': {
                        'x': pt2d[0],
                        'y': pt2d[1],
                        'z': 0
                    }
                })
        return result
