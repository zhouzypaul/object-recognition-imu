import math
import numpy as np
from pyquaternion import Quaternion
from config import *


def compute_displacement_quaternion(q: Quaternion):
    """
    compute the displacement dx, dy for one object
    input: a quaternion representing the rotation between two frames
    output: the displacement between two frames: dx, dy in pixel units
    """
    inertial_z = Quaternion(0, 0, 0, 1)  # z axis in the inertial frame
    body_z = q * inertial_z * q.inverse  # z axis rotated to the body frame, new rotate to old
    assert math.isclose(body_z.w, 0, abs_tol=1e-9), 'quaternion real part not 0'
    vector = np.array(body_z.axis)
    xz_projection = np.array([vector[0], 0, vector[2]])  # vector projected on the xz plane
    # x_angle = math.degrees(math.acos(np.dot(xz_projection, np.array([0, 0, 1])) / np.linalg.norm(xz_projection))) \
    #     if xz_projection[0] >= 0 \
    #     else - math.degrees(math.acos(np.dot(xz_projection, np.array([0, 0, 1])) / np.linalg.norm(xz_projection)))
    # y_angle = math.degrees(math.acos(np.dot(vector, xz_projection) / np.linalg.norm(xz_projection))) \
    #     if vector[1] >= 0 else - math.degrees(math.acos(np.dot(vector, xz_projection) / np.linalg.norm(xz_projection)))
    x_angle = math.degrees(math.atan(xz_projection[0] / abs(xz_projection[2])))
    y_angle = math.degrees(math.atan(vector[1] / np.linalg.norm(xz_projection)))
    dx = - x_angle / width_angle * pixel_width
    dy = y_angle / height_angle * pixel_height
    return dx, dy


def compute_displacement_pr(vx: float, vy: float, vz: float,
                            d_center: float,
                            angle: float,
                            delta_t: float = dt) -> ():
    """
    compute the displacement dx, dy for one object
    this method use the proportion of the angle turned to the angle_of_view to compute dx, dy
    input: vx, vy, vz: the angular rotational speed around x, y, z axis, obtained from imu, in degree/s
           d_center: the distance between center of object bounding box and image center
           angle: in radian, polar angle of the center of object bounding box, with respect to image center as origin
    output: the displacement between two IMU measurements: dx, dy in pixel units
    """
    # the gyro influence
    dy = - (vx * delta_t) / height_angle * pixel_height
    dx = - (vy * delta_t) / width_angle * pixel_width
    dx += d_center * (
                math.cos(angle + math.radians(vz * delta_t)) - math.cos(angle))  # rotation around z --> frame rotation
    dy += d_center * (
                math.sin(angle + math.radians(vz * delta_t)) - math.sin(angle))  # rotation around z --> frame rotation
    return dx, dy


def compute_displacement(v_x: float, v_y: float, v_z: float,
                         d_center: float,
                         angle: float,
                         depth: float = default_depth) -> ():
    """
    compute the displacement dx, dy for one object
    input: v_x, v_y, v_z: the angular rotational speed around x, y, z axis, obtained from imu, in degree/s
           d_center: the distance between center of object bounding box and image center
           angle: in radian, polar angle of the center of object bounding box, with respect to image center as origin
           depth: the distance of the object to the camera
    output: the displacement needed: dx, dy in pixel units
    """
    vx_rad, vy_rad, vz_rad = math.radians(v_x), math.radians(v_y), math.radians(v_z)  # convert degree to radian
    dy = vz_rad * dt * depth  # caused by rotation around z axis
    dx = - vx_rad * dt * depth  # caused by rotation around x axis
    dx -= d_center * (math.cos(angle) - math.cos(angle + vy_rad * dt))  # rotation around y, this is the frame rotating
    dy += d_center * (math.sin(angle + vy_rad * dt) - math.sin(angle))  # rotation around y, this is the frame rotating
    return meter_to_pixel(dx), meter_to_pixel(dy)


def compute_displacement_ls(v_x: float, v_y: float, v_z: float, n: int, depth_ls: []):
    """
    compute the displacement dx, dy
    input: v_x, v_y, v_z: the angular rotational speed around x, y, z axis, obtained from imu
           n: the number of objects that need to be moved by dx, dy
           depth_ls: a list of depth of the objects
    output: a list of displacement tuples, for how far each object has to be moved
            [(dx1, dy1), (dx2, dy2), ... ]
    """
    displacement_ls = []
    # x_roll =
    # y_pitch =
    # z_yaw =
    return displacement_ls


def meter_to_pixel(m: float) -> float:
    """
    convert meters to pixels
    input: m: in meters
    output: m in pixel units
    """
    meter_width = 2 * focus * math.tan(math.radians(width_angle) / 2)  # the width of a single pic in meters
    m_px_ratio = pixel_width / meter_width
    return m * m_px_ratio
    # TODO: different for different objects?
