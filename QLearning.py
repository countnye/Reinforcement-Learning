import random as r

import chess

import StateSpace as stateSpace
import ChessBoard as cb


class QLearning:
    """
    Q-Learning class
    """

    def __init__(self, alpha, gamma, epsilon, chess_board):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.chess_board = chess_board
        # self.chess_board = cb.ChessBoard(cb.Scenario.KKR)
        self.state_space_model = stateSpace.StateSpace(cb.Scenario.KKR)
        self.state_space = self.state_space_model.load('state_spaceKKR.pkl')
        self.end_loop = False
        self.new_board = chess_board

    def learn(self):
        """
        Function to implement Q-Learning.
        """
        for _ in range(100):
            while not self.end_loop:
                curr_state = self.chess_board.get_board_representation()
                if r.uniform(0, 1) < self.epsilon:
                    move = self.chess_board.get_random_move()
                else:
                    _, move = self.get_next(self.chess_board.copy())
                # get Q(s_t,a_t)
                curr_val = self.get_q_val(curr_state, move)
                # get max Q(s_{t+1}, a)
                next_state = self.chess_board.copy()
                next_state.make_move(move)
                next_state.switch_turns()
                next_q_val, _ = self.get_next(next_state)
                # if the move results in checkmate, reward is 100
                if next_state.is_checkmate():
                    reward = 100
                # if the move results in stalemate, reward is 0
                elif next_state.is_stalemate():
                    reward = 0
                # if simply move made, reward is -1
                else:
                    reward = -1
                # calculate TD error
                td_error = reward + self.gamma * next_q_val - curr_val
                # update Q value
                self.state_space[curr_state][move] = curr_val + self.alpha * td_error
                # make the move
                self.chess_board.make_move(move)
                # end loop if checkmate
                if self.chess_board.is_checkmate():
                    self.end_loop = True
            # reset the board after each epoch
            self.chess_board.reset()
            self.end_loop = False

    def get_q_val(self, state, action):
        """
        Function to get the Q-Value of the given state.
        :param state: the representation of current state
        :param action: the action from current state
        :return: the Q-Value of the current state-action pair
        """
        # if the action or state was missing from state space, it was not explored
        return self.state_space[state][action]

    def get_next(self, board):
        """
        Function to get the max/min Q-Value of a given state.
        :param board: the board representation of the current state
        :return: the max/min q-value and the corresponding action
        """
        board_rep = board.get_board_representation()
        action_set = board.get_legal_moves()
        best_action = action_set[0]
        q_val = self.get_q_val(board_rep, best_action)
        for action in action_set:
            curr_q_val = self.get_q_val(board_rep, action)
            if board.get_turn() == 'WHITE' and curr_q_val > q_val:
                q_val = curr_q_val
                best_action = action
            elif board.get_turn() == 'BLACK' and curr_q_val < q_val:
                q_val = curr_q_val
                best_action = action
        return q_val, best_action

    def test_board(self):
        """
        Function to return the best actions taken using Q values.
        :return: the list of actions taken.
        """
        best_actions = []
        run = True
        while run:
            _, action = self.get_next(self.chess_board)
            best_actions.append(action)
            self.chess_board.make_move(action)
            if self.chess_board.is_stalemate() or self.chess_board.is_checkmate():
                if self.chess_board.is_checkmate():
                    print(self.chess_board.get_turn() + ' Checkmated!')
                else:
                    print(self.chess_board.get_turn() + ' Stalemated!')
                run = False
        return best_actions
