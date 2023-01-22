import chess
from enum import Enum
import random as r
import copy


class GameState(Enum):
    RUNNING = 'GAME NOT OVER'
    IN_CHECK_BLACK = 'BLACK KING IN CHECK'
    CHECKMATE_BLACK = 'WHITE WON'
    STALEMATE = 'GAME IN DRAW'


class ChessBoard:
    def __init__(self, board_type):
        # initialize the given board
        self.board = chess.Board()
        self.board_type = board_type
        if self.board_type == 'KKQ':
            self.init_board_KKQ()
        else:
            self.init_board_KKR()
        # define a state to check end of game
        self.game_state = GameState.RUNNING

    def init_board_KKQ(self):
        """
        Function to initialize the King vs King and Queen endgame.
        """
        self.board.clear_board()
        self.board.set_piece_at(chess.C3, chess.Piece(chess.KING, chess.WHITE))
        self.board.set_piece_at(chess.G3, chess.Piece(chess.QUEEN, chess.WHITE))
        self.board.set_piece_at(chess.C5, chess.Piece(chess.KING, chess.BLACK))

    def init_board_KKR(self):
        """
        Function to initialize the King vs King and Rook endgame.
        """
        self.board.clear_board()
        self.board.set_piece_at(chess.D2, chess.Piece(chess.KING, chess.WHITE))
        self.board.set_piece_at(chess.G3, chess.Piece(chess.ROOK, chess.WHITE))
        self.board.set_piece_at(chess.C5, chess.Piece(chess.KING, chess.BLACK))

    def make_move(self, move):
        """
        Function to make a move.
        :param move: the move to be made
        """
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

    def get_random_move(self):
        """
        Function to get a random move from the possible legal moves.
        :return: the random move in UCI format
        """
        possible_moves = self.get_legal_moves()
        random_idx = r.randint(0, len(possible_moves) - 1)
        return possible_moves[random_idx]

    def get_legal_moves(self):
        """
        Function to get all possible legal moves for the current player.
        :return: the list of all possible legal moves
        """
        legal_moves = []
        for move in self.board.legal_moves:
            legal_moves.append(str(move))
        return legal_moves

    def is_checkmate(self):
        """
        Function to check if board is in checkmate.
        :return: True if checkmate, else False
        """
        return self.board.is_checkmate()

    def get_turn(self):
        """
        Function to get the player who is currently playing.
        :return: 'WHITE' if white's turn, else 'BLACK'
        """
        return 'WHITE' if self.board.turn == True else 'BLACK'

    def get_board_representation(self):
        """
        Function to get the FEN representation of the board.
        :return: FEN representation of the board
        """
        return self.board.fen()

    def pop(self):
        """
        Function to undo a move.
        """
        self.board.pop()

    def reset(self):
        """
        Function to reset board.
        """
        if self.board_type == 'KKQ':
            self.init_board_KKQ()
        else:
            self.init_board_KKR()

    def print_board(self):
        """
        Function to print the board.
        """
        print(self.board)
        print('==============')

    def copy(self):
        """
        Function to get the copy of this ChessBoard object.
        :return: the copy of this instance of the ChessBoard object
        """
        return ChessBoard(copy.copy(self.board_type))


# board = ChessBoard('KKR')
# print('OG Board:')
# board.print_board()
# board1 = board.copy()
# print('Board1:')
# board1.print_board()
# board1.make_move(board1.get_legal_moves()[0])
# print('Board1 after move:')
# board1.print_board()
# print('OG Board after move:')
# board.make_move(board.get_random_move())
# board.print_board()
# print(board.get_board_representation())
# # makes random legal moves until someone is under check
# while not board.board.is_check():
#     legal_move = board.get_legal_moves()
#     curr_move = legal_move[r.randint(0, len(legal_move) - 1)]
#     board.make_move(curr_move)
#     print('move made = ', curr_move)
#     board.print_board()
#     board.board.is_fivefold_repetition()

# (!1)
