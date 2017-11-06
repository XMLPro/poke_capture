from ctypes import *
import numpy
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

class Cell(Structure):
    _fields_ = [
            ("x", c_int),
            ("y", c_int),
            ("is_alive", c_bool),
            ("r", c_int),
            ("g", c_int),
            ("b", c_int),
            ]

    def __init__(self, x, y, is_alive, r, g, b):
        self.x = x
        self.y = y
        self.is_alive = is_alive
        self.r = r
        self.g = g
        self.b = b


class GameMap(Structure):
    _fields_ = [
            ("width", c_int),
            ("height", c_int),
            ("cells", POINTER(POINTER(Cell))),
            ]

    def __init__(self):
        pass

lg = CDLL(base_dir + "/lg.so")

def blur(data, n):
    game_map = GameMap()
    lg.init_game_map(byref(game_map), 28, 28)
    ndata = (data * 255).astype(numpy.int32).tolist()
    for i in range(28):
        for j, v in enumerate(ndata[i*28 : i*28 + 28]):
            game_map.cells[i][j].r = v

    for i in range(13):
        if n & 1:
            lg.random_spawn(byref(game_map), 40)
        n >>= 1
        lg.next(byref(game_map))

    after = [0 for x in range(28 * 28)]
    for i in range(28):
        for j in range(28):
            after[i*28 + j] = game_map.cells[i][j].r

    after = numpy.array(after, dtype=numpy.float32)
    return after

