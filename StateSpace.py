class StateSpace:
    def __init__(self, board):
        self.board = board
        self.state_space = {}
        self.set_state_space()

    # function to initialize the state space
    def set_state_space(self):
        state_space = {}
        # create a function that takes self.board and applies all possible legal
        # moves and stores all those new board in the coordinate representation.
        # check for repeating boards and when no new board is possible, end loop.
        self.state_space = state_space

    # function to get state space
    def get_state_space(self):
        return self.state_space
