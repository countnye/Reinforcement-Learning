import chess

import ChessBoard as cb
import QLearningTest as qlt
import QLearning as ql
import time
import matplotlib.pyplot as plt


def plot(stats):
    print("wins: ", stats['wins'])
    print("game_length: ", stats['game_length'])
    print("games_played: ", stats['games_played'])


def plot_win_rates(win_rate1, epochs1):
    x = [i for i in range(0, epochs1+1)]
    labels = ['0.9997', '0.95', '0.90', '0.80', '0.70', '0.60', '0.50']
    for i, y in enumerate(win_rate1):
        plt.plot(x, y, label=labels[i])
    plt.xlabel('Number of Epochs')
    plt.ylabel('Win Rate (%)')
    plt.title('Win rate over Epochs of Models with different Decay rates')
    plt.legend()
    plt.show()


# for training run this. This will save the statespace.
# NOTE: if u ran this once, i.e. the win rate graph printed, you can remove the saving file part
epochs = 300000
e_decays = [0.9997, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50]
win_rates = []
start_time = time.time()
save_file = True
for e_decay in e_decays:
    scenario = cb.Scenario.KKR
    chess_board = cb.ChessBoard(scenario)
    model = ql.QLearning(epochs, 0.9, 0.99, 1.0, e_decay, chess_board)
    model.learn()
    win_rates.append(model.win_rate)
    if save_file:
        # change the file name to show which endgame it is, i.e. KKR or KKQ
        model.save('updated_q_tableKKQ.pkl')
        save_file = False
print("--- %s seconds ---" % (time.time() - start_time))
plot_win_rates(win_rates, epochs)

# # for testing run this, this loads the saved statespace and prints win rate plot
# epochs = 300000
# e_decays = [0.9997, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50]
# win_rates = []
# start_time = time.time()
# scenario = cb.Scenario.KKR
# chess_board = cb.ChessBoard(scenario)
# model = ql.QLearning(epochs, 0.9, 0.99, 1.0, 0.9997, chess_board)
# win_rates.append(model.test_learning(epochs))
# print("--- %s seconds ---" % (time.time() - start_time))
# plot_win_rates(win_rates, epochs)


