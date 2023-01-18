import chess
from enum import Enum
import random as r


class GameState(Enum):
    RUNNING = 'GAME NOT OVER'
    IN_CHECK_BLACK = 'BLACK KING IN CHECK'
    CHECKMATE_BLACK = 'WHITE WON'
    STALEMATE = 'GAME IN DRAW'


class ChessBoard:
    def __init__(self, board_type):
        # initialize the given board
        self.board = chess.Board()
        if board_type == 'KKQ':
            self.init_board_KKQ()
        else:
            self.init_board_KKR()
        # define a state to check end of game
        self.game_state = GameState.RUNNING

    # function to initialize board with King vs King & Queen endgame
    def init_board_KKQ(self):
        self.board.clear_board()
        self.board.set_piece_at(chess.C3, chess.Piece(chess.KING, chess.WHITE))
        self.board.set_piece_at(chess.G3, chess.Piece(chess.QUEEN, chess.WHITE))
        self.board.set_piece_at(chess.C5, chess.Piece(chess.KING, chess.BLACK))

    # function to initialize board with King vs King & Rook endgame
    def init_board_KKR(self):
        self.board.clear_board()
        self.board.set_piece_at(chess.D2, chess.Piece(chess.KING, chess.WHITE))
        self.board.set_piece_at(chess.G3, chess.Piece(chess.ROOK, chess.WHITE))
        self.board.set_piece_at(chess.C5, chess.Piece(chess.KING, chess.BLACK))

    # function to make a move
    def make_move(self, move):
        # only make a move if game has not ended
        if self.game_state == GameState.RUNNING:
            move = chess.Move.from_uci(move)
            if move in self.board.legal_moves:
                self.board.push(move)
                if self.board.is_check():
                    print(self.get_turn() + ' is in check!')
            else:
                print('Illegal move.')
        else:
            print('Game is not active.')

    # function to get possible legal moves
    def get_legal_moves(self):
        legal_moves = []
        for move in self.board.legal_moves:
            legal_moves.append(str(move))
        return legal_moves

    # function to get turn
    def get_turn(self):
        return 'WHITE' if self.board.turn == True else 'BLACK'

    # function to print the board
    def print_board(self):
        print(self.board)
        print('==============')


board = ChessBoard('KKR')
board.print_board()
# # makes random legal moves until someone is under check
# while not board.board.is_check():
#     legal_move = board.get_legal_moves()
#     curr_move = legal_move[r.randint(0, len(legal_move) - 1)]
#     board.make_move(curr_move)
#     print('move made = ', curr_move)
#     board.print_board()

# (!1)
