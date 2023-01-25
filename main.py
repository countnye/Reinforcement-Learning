import ChessBoard as cb
import QLearning as ql
import time

def plot(stats):
    print("wins: ", stats['wins'])
    print("game_length: ", stats['game_length'])
    print("games_played: ", stats['games_played'])

epochs = 100
start_time = time.time()

scenario = cb.Scenario.KKR
chess_board = cb.ChessBoard(scenario)
model = ql.QLearning(epochs, 0.5, 0.9, 0.2, chess_board)
stats_q_learning = model.get_stats()
plot(stats_q_learning)

print("--- %s seconds ---" % (time.time() - start_time))

best_moves = model.test_board()
print(best_moves)
