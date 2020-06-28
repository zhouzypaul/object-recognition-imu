import math
from config import *


def compute_displacement_pr(vx: float, vy: float, vz: float,
                            d_center: float,
                            angle: float) -> ():
    """
    compute the displacement dx, dy for one object
    this method use the proportion of the angle turned to the angle_of_view to compute dx, dy
    input: vx, vy, vz: the angular rotational speed around x, y, z axis, obtained from imu, in degree/s
           d_center: the distance between center of object bounding box and image center
           angle: in radian, polar angle of the center of object bounding box, with respect to image center as origin
    output: the displacement needed: dx, dy in pixel units
    """
    # the gyro influence
    dy = - (vx * dt) / height_angle * pixel_height
    dx = - (vy * dt) / width_angle * pixel_width
    dx += d_center * (math.cos(angle + math.radians(vz * dt)) - math.cos(angle))  # rotation around z --> frame rotation
    dy += d_center * (math.sin(angle + math.radians(vz * dt)) - math.sin(angle))  # rotation around z --> frame rotation
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
    dx = - vx_rad * dt * depth   # caused by rotation around x axis
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
