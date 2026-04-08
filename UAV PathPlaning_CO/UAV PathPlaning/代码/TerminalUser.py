

class TerminalUser:
    def __init__(self, id, x, y, I):
        self.id = id
        self.x = x
        self.y = y
        self.flag_done = False
        self.I = I
        self.I_origin = I
        self.r = 0.05