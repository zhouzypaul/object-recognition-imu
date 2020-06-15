def move_objects(objs: [], dx: float, dy: float):
    """
    update the bounding box location of the object
    input: objs: [('tag', con, (x, y, w, h))]
           dx: delta x
           dy: delta y
    output: None, update the original obj
    """
    # TODO: if objs is NONE
    if objs is None:
        return []
    moved_objects = []
    for obj in objs:
        new_x = obj[2][0] + dx
        new_y = obj[2][1] + dy
        new_obj = (obj[0], obj[1], (new_x, new_y, obj[2][2], obj[2][3]))
        moved_objects.append(new_obj)
    return moved_objects
