import chess

import ChessBoard as cb
import QLearning as ql
import time


def plot(stats):
    print("wins: ", stats['wins'])
    print("game_length: ", stats['game_length'])
    print("games_played: ", stats['games_played'])


# if u want to test the output for multiple epochs, add them to the array below
# else just add the one epoch
epochs = [100]
for epoch in epochs:
    print('=====', epoch, '=====')
    start_time = time.time()
    scenario = cb.Scenario.KKR
    chess_board = cb.ChessBoard(scenario)
    model = ql.QLearning(epoch, 0.1, 0.9, 1.0, chess_board)
    model.learn()
    stats_q_learning = model.get_stats()
    plot(stats_q_learning)
    print("--- %s seconds ---" % (time.time() - start_time))

# best_moves = model.test_board()
# print(best_moves)
