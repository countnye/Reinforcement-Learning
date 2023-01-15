import chess


# function to initialize board with King vs King & Queen endgame
def init_board_KKQ():
    KKQ_board = chess.Board()
    KKQ_board.clear_board()
    KKQ_board.set_piece_at(chess.C3, chess.Piece(chess.KING, chess.WHITE))
    KKQ_board.set_piece_at(chess.G3, chess.Piece(chess.QUEEN, chess.WHITE))
    KKQ_board.set_piece_at(chess.C5, chess.Piece(chess.KING, chess.BLACK))
    return KKQ_board


# function to initialize board with King vs King & Rook endgame
def init_board_KKR():
    KKR_board = chess.Board()
    KKR_board.clear_board()
    KKR_board.set_piece_at(chess.D2, chess.Piece(chess.KING, chess.WHITE))
    KKR_board.set_piece_at(chess.G3, chess.Piece(chess.ROOK, chess.WHITE))
    KKR_board.set_piece_at(chess.C5, chess.Piece(chess.KING, chess.BLACK))
    return KKR_board


class ChessBoard:
    def __init__(self, board_type):
        # initialize the given board
        if board_type == 'KKQ':
            self.board = init_board_KKQ()
        else:
            self.board = init_board_KKR()

    # function to print the board
    def print_board(self):
        print(self.board)


board = ChessBoard('KKR')
board.print_board()

# (!1)
