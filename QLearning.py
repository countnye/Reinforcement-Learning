import random as r

import StateSpace as stateSpace


class QLearning:
    """
    Q-Learning class
    """
    def __init__(self, alpha, gamma, epsilon, chess_board):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.chess_board = chess_board
        self.state_space = stateSpace.StateSpace(self.chess_board).get_state_space()
        self.end_loop = False

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
                next_q_val = self.get_next(next_state)
                # -1 reward for making move
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
        :param state: the FEN of current state
        :param action: the action from current state
        :return: the Q-Value of the current state-action pair
        """
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

