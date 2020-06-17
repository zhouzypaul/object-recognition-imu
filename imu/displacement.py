import math


# TODO: make sure the following parameters are correct before running main
default_depth = 3  # in meters
dt = 0.1  # in seconds, the time interval between two frames
view_angle = 100  # in degrees, the angle of view from the RGB camera
pixel_width = 500  # the length of a single picture, in pixel units
focus = 0.01  # the distance between the camera eye and the screen where picture is formed. in meters


def compute_displacement_pr(vx: float, vy:float, vz: float,
                            d_center: float,
                            angle: float) -> ():
    """
    compute the displacement dx, dy for one object
    this method use the proportion of the angle turned to the angle_of_view to compute dx, dy
    input: v_x, v_y, v_z: the angular rotational speed around x, y, z axis, obtained from imu, in degree/s
           d_center: the distance between center of object bounding box and image center
           angle: in radian, polar angle of the center of object bounding box, with respect to image center as origin
    output: the displacement needed: dx, dy in pixel units
    """
    dy = (vz * dt) / view_angle * pixel_width
    dx = - vx * dt / view_angle * pixel_width  # TODO: is the view angle the same for x and y???
    dx -= d_center * (math.cos(angle) - math.cos(angle + math.radians(vy) * dt))  # rotation around y --> frame rotation
    dy += d_center * (math.sin(angle + math.radians(vy) * dt) - math.sin(angle))  # rotation around y --> frame rotation
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
    dy = vz_rad * dt * depth  # caused by rotation around z axis  TODO: verify the angle vz*dt is small
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
    meter_width = 2 * focus * math.tan(math.radians(view_angle) / 2)  # the width of a single pic in meters
    m_px_ratio = pixel_width / meter_width
    return m * m_px_ratio
    # TODO: different for different objects?
