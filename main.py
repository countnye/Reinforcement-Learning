import ChessBoard as cb
import QLearning as ql
import StateSpace as sp

scenario = cb.Scenario.KKR
state_space_class = sp.StateSpace(scenario)
state_space_class.save("state_space.pkl")

chess_board = cb.ChessBoard(scenario)
model = ql.QLearning(0.5, 0.9, 0.2, chess_board)
model.learn()
