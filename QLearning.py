import numpy as np
import random as r

import ChessBoard as chessBoard
import StateSpace as stateSpace


class QLearning:
    def __init__(self, alpha, gamma, epsilon, chess_board):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # self.chess_board = chess_board
        self.chess_board = chessBoard.ChessBoard('KRK')
        # define the state space
        self.state_space = stateSpace.StateSpace(self.chess_board).get_state_space()
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

    # function to get the max Q value given a set of moves
    def get_next(self, action_set, turn):
        self.chess_board.make_move(action_set[0])
        board_rep = self.chess_board.get_board_representation()
        next_q = self.state_space[board_rep]
        self.chess_board.board.pop()
        next_action = action_set[0]
        for move in action_set:
            self.chess_board.make_move(move)
            board_rep = self.chess_board.get_board_representation()
            # for each action get its Q value
            if turn == 'WHITE' and self.state_space[board_rep] > next_q:
                next_q = self.state_space[board_rep]
                next_action = move
            elif turn == 'BLACK' and self.state_space[board_rep] < next_q:
                next_q = self.state_space[board_rep]
                next_action = move
            # undo the move from the board
            self.chess_board.board.pop()
        return next_q, next_action

