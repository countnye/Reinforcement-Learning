import chess

'''''

Might need to copy the board to prevent breaking. Keyword: might
There's an inbuilt copy function in the chess module.

'''''

class StateSpace:
    def __init__(self, board):
        self.board = board.copy()
        self.state_space = {}
        self.set_state_space()

    # function to initialize the state space
    def set_state_space_rec(self):
        # create a function that takes self.board and applies all possible legal
        # moves and stores all those new board in the coordinate representation.
        # check for repeating boards and when no new board is possible, end loop.

        # state space should be a dict in a dict in form {'state': {'action': q-value} }
        if self.board.is_game_over():
            return


        for move in self.board.get_legal_moves():
            self.board.push(move)
            self.set_state_space_rec()
            action = {}
            # get FEN representation of board
            state = self.board.fen()
            # if state is not already in dict, add it
            # and init q-value to 0
            if state not in self.state_space.keys():
                self.state_space[state] = {move : 0}
            else:

            self.board.pop()




    # function to get state space
    def get_state_space(self):
        return self.state_space
