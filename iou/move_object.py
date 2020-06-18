def move_object(obj: (), dx: float, dy: float) -> ():
    """
    change the bounding box location of an object
    input: obj: an object, ('tag', con, (x, y, w, h))
           dx: delta x
           dy: delta y
    output: a new object with the new location
    """
    tag, con, x, y, w, h = obj[0], obj[1], obj[2][0], obj[2][1], obj[2][2], obj[2][3]
    return tag, con, (x + dx, y + dy, w, h)


def move_objects(objs: [], dx: float, dy: float) -> []:
    """
    change the bounding box location of a list of objects
    input: objs: [('tag', con, (x, y, w, h))]
           dx: delta x
           dy: delta y
    output: a new list of objects with the new location
    """
    if objs is None:
        return []
    moved_objects = []
    for obj in objs:
        new_x = obj[2][0] + dx
        new_y = obj[2][1] + dy
        new_obj = (obj[0], obj[1], (new_x, new_y, obj[2][2], obj[2][3]))
        moved_objects.append(new_obj)
    return moved_objects
