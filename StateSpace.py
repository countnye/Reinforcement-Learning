import chess
import pickle
import ChessBoard as cb

'''''

Might need to copy the board to prevent breaking. Keyword: might
There's an inbuilt copy function in the chess module.

'''''

class StateSpace:
    def __init__(self, board):
        self.board = board.copy()
        self.state_space = {}
        self.set_state_space_rec()
        self.i = 0

    # function to initialize the state space
    def set_state_space_rec(self):
        # create a function that takes self.board and applies all possible legal
        # moves and stores all those new board in the coordinate representation.
        # check for repeating boards and when no new board is possible, end loop.

        # state space should be a dict in a dict in form {'state': {'action': q-value} }

        # ASSUMPTION: this block does not fire if game is over
        for move in self.board.get_legal_moves():
            self.board.push(move)
            self.set_state_space_rec()
            # get FEN representation of board
            state = self.board.fen()
            # if state is not already in dict, add it
            # and init q-value to 0
            if state not in self.state_space.keys():
                self.state_space[state] = {move : 0}
            else:
                # init's Q-value of state-action cell to 0
                self.state_space[state][move] = 0
            self.board.pop()

        return

    
    def save(self, filename):
        '''''
        Purpose of this function is to save state space,
        avoiding re-computation
        '''''
        with open(filename, 'wb') as f:
            pickle.dump(self.state_space, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.state_space = pickle.load(f)
            return self.state_space


    # function to get state space
    def get_state_space(self):
        return self.state_space


board = cb.ChessBoard('KKR')
test_space = StateSpace(board.board)
test_space.set_state_space_rec()
print("done")
test_space.save("state_space.bson")

