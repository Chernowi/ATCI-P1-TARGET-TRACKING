from world_objects import Object, Location

def calculate_distance(obj1: Object, obj2: Object) -> float:
    """
    Calculate the Euclidean distance between two objects in 3D space.
    
    Args:
        obj1 (Object): The first object.
        obj2 (Object): The second object.
        
    Returns:
        float: The distance between the two objects.
    """
    return ((obj1.location.x - obj2.location.x) ** 2 +
            (obj1.location.y - obj2.location.y) ** 2 +
            (obj1.location.depth - obj2.location.depth) ** 2) ** 0.5