

class TerminalUser:
    def __init__(self, id, x, y, I):
        self.id = id
        self.x = x
        self.y = y
        self.flag_done = False
        self.I = I
        self.I_origin = I
        self.r = 0.05
        self.event_type = 'default'
        self.event_priority = I
        self.spawn_time = 0
        self.active = True
        self.decay_rate = 0.0
        self.last_decay_step = 0
