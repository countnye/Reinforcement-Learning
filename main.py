import ChessBoard as cb
import QLearning as ql

chess_board = cb.ChessBoard('KKR')
model = ql.QLearning(0.5, 0.9, 0.2, chess_board)
model.learn()
