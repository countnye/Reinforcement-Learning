import numpy as np
import random as r

import chess_board as chessBoard


class QLearning:
    def __init__(self, alpha, gamma, epsilon, chess_board):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # self.chess_board = chess_board
        self.chess_board = chessBoard.ChessBoard('KRK')
        self.end_loop = False

    def learn(self):
        while not self.end_loop:
            # epsilon chance of exploration
            if r.uniform(0, 1) < self.epsilon:
                move = self.chess_board.get_random_move()
            else:
                # for all moves get move with max reward
                pass
            # change this to previous q_val once state space is defined
            prev_q_val = 0
            # change this to maximum/minimum possible Q value for the next state
            if self.chess_board.get_turn() == 'WHITE':
                # get the max Q value for the next state
                next_val = 10
            else:
                # get the min Q value for the next state
                next_val = 0
            # change this to reward for the given move
            reward = 10
            # calculate the new Q value
            new_q_val = (1 - self.alpha) * prev_q_val + self.alpha * (reward + self.gamma * next_val)

