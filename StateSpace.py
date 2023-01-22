import pickle
import ChessBoard as cb


class StateSpace:
    """
    StateSpace class
    """
    def __init__(self, board):
        # self.board = board.copy()
        self.chess_board = cb.ChessBoard('KKR')
        self.state_space = {}
        self.set_state_space_rec()
        self.i = 0

    def set_state_space_rec(self):
        """
        Function to generate the state-space
        :return: the state-space dictionary
        """
        # state space should be a dict in a dict in form {'state': {'action': q-value} }

        # ASSUMPTION: this block does not fire if game is over
        for move in self.chess_board.get_legal_moves():
            self.chess_board.board.push(move)
            self.set_state_space_rec()
            # get FEN representation of board
            state = self.chess_board.get_board_representation()
            # if state is not already in dict, add it
            # and init q-value to 0
            if state not in self.state_space.keys():
                self.state_space[state] = {move: 0}
            else:
                # inits Q-value of state-action cell to 0
                self.state_space[state][move] = 0
            self.chess_board.pop()

        return

    def save(self, filename):
        """
        Function to save state space, avoiding re-computation.
        :param filename: the name file will be saved as
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.state_space, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        """
        Function to load the state-space file
        :param filename: the name of the file to load
        :return: the state-space dictionary
        """
        with open(filename, 'rb') as f:
            self.state_space = pickle.load(f)
            return self.state_space

    def get_state_space(self):
        """
        Function to retrieve the state-space
        :return: the state-space dictionary
        """
        return self.state_space


board = cb.ChessBoard('KKR')
test_space = StateSpace(board.board)
test_space.set_state_space_rec()
print("done")
test_space.save("state_space.bson")
