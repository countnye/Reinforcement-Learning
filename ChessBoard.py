import chess
from enum import Enum
import random as r
import copy


class Scenario(Enum):
    KKQ = 0
    KKR = 1


class GameState(Enum):
    RUNNING = 'GAME NOT OVER'
    IN_CHECK_BLACK = 'BLACK KING IN CHECK'
    CHECKMATE_BLACK = 'WHITE WON'
    STALEMATE = 'GAME IN DRAW'


class ChessBoard:
    def __init__(self, board_type, board=chess.Board(), isNewBoard = True):
        # initialize the given board
        self.board_type = board_type
        self.board = board
        if self.board_type == Scenario.KKQ and isNewBoard:
            self.init_board_KKQ()
        elif self.board_type == Scenario.KKR and isNewBoard:
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
                print('Illegal move.:', move, " current turn: ", self.get_turn(), "board rep: ", self.get_board_representation())
        else:
            print('Game is not active.')

    def switch_turns(self):
        """
        Function to switch turns without making a move.
        """
        self.board.push(chess.Move.null())

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

    def is_stalemate(self):
        """
        Function to check if board is in stalemate.
        :return: True is stalemate, else False
        """
        
        return True if self.board.is_stalemate() or self.board.is_fifty_moves() or self.board.is_insufficient_material() else False

    def get_turn(self):
        """
        Function to get the player who is currently playing.
        :return: 'WHITE' if white's turn, else 'BLACK'
        """
        return 'WHITE' if self.board.turn == chess.WHITE else 'BLACK'

    def get_board_representation(self):
        """
        Function to get the representation of the board.
        The representation is in the form (bk_square, wk_square, wp_square).
        :return: the representation of the board
        """
        bk_pos = self.board.king(chess.BLACK)
        wk_pos = self.board.king(chess.WHITE)
        # if there is no other non-king white piece on board, set pos to -1
        if len(self.board.pieces(chess.ROOK if self.board_type == Scenario.KKR else chess.QUEEN, chess.WHITE)) == 0:
            wp_pos = None
        else:
            wp_pos = list(self.board.pieces(chess.ROOK if self.board_type == Scenario.KKR else chess.QUEEN, chess.WHITE))[0]
        return (bk_pos, wk_pos, wp_pos)

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
        #BUG: BECAUSE THIS RETURNS SHALLOW COPY, A MOVE IS STILL MADE ON 
        #     OG TABLE
        """
        Function to get the copy of the board object.
        :return: the copy of this instance of the board object
        """
        # return ChessBoard(copy.copy(self.board_type))
        return copy.copy(self)

    def copy_board(self, state, turn):
        """
        Returns a chess.Board object with the set state
        """
        board = chess.Board()
        board.clear()

        board.set_piece_at(state[0], chess.Piece(chess.KING, chess.BLACK))
        board.set_piece_at(state[1], chess.Piece(chess.KING, chess.WHITE))
        if self.board_type == Scenario.KKQ and state[2] != None:
            board.set_piece_at(state[2], chess.Piece(chess.QUEEN, chess.WHITE))
        elif self.board_type == Scenario.KKR and state[2] != None:
            board.set_piece_at(state[2], chess.Piece(chess.ROOK, chess.WHITE))
        
        new = ChessBoard(self.board_type, board, False)
        new.board.turn = turn
        return new


# board = ChessBoard('KKR')
# print(board.get_board_representation())
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
