import ChessBoard as cb
import QLearning as ql

scenario = cb.Scenario.KKR
chess_board = cb.ChessBoard(scenario)
model = ql.QLearning(0.5, 0.9, 0.2, chess_board)
model.learn()
best_moves = model.test_board()
print(best_moves)
