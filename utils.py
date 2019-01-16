def decode_pos(pos, size, offset=1):
    """
    transforms integer position on grid into i, j
    """
    return (pos - offset) // size, (pos - offset) % size

def encode_pos(i, j, size, offset=1):
    """
    encode i, j position into integer (with offset for additionnal information
    e.g. total absence)
    """
    pos = offset + j + i * size
    if pos > size ** 2:
        return 0
    else:
        return pos


def move_pos(coord, dir, size):
    if dir == 1:
        new_coord = max(0, coord[0] - 1), coord[1]
    elif dir == 2:
        new_coord = min(size - 1, 1 + coord[0]), coord[1]
    elif dir == 3:
        new_coord = coord[0], min(size - 1, coord[1] + 1)
    elif dir == 4:
        new_coord = coord[0], max(0, coord[1] - 1)
    else:
        new_coord = coord
    return new_coord, (new_coord == coord)
