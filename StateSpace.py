import pickle
import ChessBoard as cb
import time
import chess


class StateSpace:
    """
    StateSpace class
    """

    def __init__(self, scenario):
        self.scenario = scenario
        self.state_space = {}

    def set_state_space(self):
        board = chess.Board()
        for bk_r in range(0, 8):
            for bk_c in range(0, 8):
                bk_pos = chess.square(bk_c, bk_r)
                for wk_r in range(0, 8):
                    for wk_c in range(0, 8):
                        wk_pos = chess.square(wk_c, wk_r)
                        for w2_r in range(-1, 8):
                            for w2_c in range(-1, 8):
                                for turn in ['BLACK', 'WHITE']:
                                    if (w2_r != -1 and w2_c == -1) or (w2_r != -1 and w2_c == -1):
                                        continue
                                    if w2_r == -1 and w2_c == -1:
                                        w2_pos = None
                                    else:
                                        w2_pos = chess.square(w2_c, w2_r)
                                    # skip overlapping positions
                                    if (bk_pos == wk_pos or
                                            bk_pos == w2_pos or
                                            wk_pos == w2_pos):
                                        continue
                                    self.create_action_pairs(bk_pos, wk_pos, w2_pos, turn)

    def create_action_pairs(self, bk_pos, wk_pos, w2_pos, turn):
        board = chess.Board()
        board.clear()
        board.set_piece_at(bk_pos, chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(wk_pos, chess.Piece(chess.KING, chess.WHITE))
        if (self.scenario == cb.Scenario.KKQ) and w2_pos is not None:
            board.set_piece_at(w2_pos, chess.Piece(chess.QUEEN, chess.WHITE))
        elif (self.scenario == cb.Scenario.KKR) and w2_pos is not None:
            board.set_piece_at(w2_pos, chess.Piece(chess.ROOK, chess.WHITE))
        actions = {}
        # get all white (current) moves
        if turn == 'WHITE':
            for move in board.legal_moves:
                actions[str(move)] = 0.0
        else:
            # change turn to black and get moves
            board.push(chess.Move.null())
            for move in board.legal_moves:
                actions[str(move)] = 0.0
        self.state_space[(bk_pos, wk_pos, w2_pos, turn)] = actions

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


# test_space = StateSpace(cb.Scenario.KKR)
# start_time = time.time()
# test_space.set_state_space()
# test_space.save("state_spaceKKR.pkl")
# print("--- %s seconds ---" % (time.time() - start_time))
