import random as r

import ChessBoard as cb
import StateSpace as st

import pickle


class QLearning:
    def __init__(self, epochs, alpha, gamma, epsilon, e_decay, chess_board):
        self.epochs = epochs
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.e_decay = e_decay
        self.board = chess_board
        self.state_space_model = st.StateSpace(cb.Scenario.KKR)
        self.state_space = self.state_space_model.load('state_spaceKKR.pkl')
        self.end_game = False
        self.win_rate = [0 for _ in range(self.epochs + 1)]

    def learn(self):
        total_wins = 0
        for epoch in range(self.epochs + 1):
            while not self.end_game:
                # get the state of the current board
                current_state = self.board.get_board_representation()
                # get the move to be made based on exploration
                if r.uniform(0, 1) < self.epsilon:
                    action = self.board.get_random_move()
                else:
                    _, action = self.get_next_val(self.board)
                # get the Q-value of the current state
                current_q_val = self.get_q_val(current_state, action)
                # make the chosen move
                self.board.make_move(action)
                # get the next state
                next_state = self.board.copy()
                next_q_val = 0
                # get the best Q-value of the next state, if next state is not stalemate
                if len(next_state.get_legal_moves()) != 0:
                    next_q_val, _ = self.get_next_val(next_state)
                # define the rewards
                if self.board.is_checkmate():
                    reward = 100
                    total_wins += 1
                elif self.board.is_draw():
                    reward = -100
                else:
                    reward = (10 - self.board.get_king_distance() - self.board.get_corner_distance()) * 10
                    if self.board.get_turn() == 'BLACK':
                        reward = (self.board.get_king_distance() + self.board.get_corner_distance()) * 10
                        reward -= 20
                # compute the td-error
                td_error = reward + (self.gamma * next_q_val) - current_q_val
                # update the Q-table
                update_val = current_q_val + (self.alpha * td_error)
                self.state_space[current_state][action] = update_val
                # check for end of game
                if self.board.is_checkmate():
                    self.end_game = True
                elif self.board.is_draw():
                    self.end_game = True
            # reset the board for next epoch
            self.board.reset()
            # decay epsilon
            self.epsilon *= self.e_decay
            self.end_game = False
            self.win_rate[epoch] = (total_wins / (epoch + 1)) * 100

    def get_q_val(self, state, action):
        return self.state_space[state][action]

    def get_next_val(self, board):
        state = board.get_board_representation()
        legal_actions = board.get_legal_moves()
        best_action = legal_actions[0]
        best_q_val = self.get_q_val(state, best_action)
        for action in legal_actions:
            q_val = self.get_q_val(state, action)
            if board.get_turn() == 'WHITE' and q_val > best_q_val:
                best_q_val = q_val
                best_action = action
            elif board.get_turn() == 'BLACK' and q_val < best_q_val:
                best_q_val = q_val
                best_action = action
        return best_q_val, best_action

    def test_learning(self, epochs):
        win_rate = [0 for _ in range(epochs + 1)]
        total_wins = 0
        board = cb.ChessBoard(cb.Scenario.KKR)
        for epoch in range(epochs + 1):
            if epoch % 50000 == 0:
                print(epoch)
            game_end = False
            while not game_end:
                q_val, action = self.get_next_val(board)
                board.make_move(action)
                if board.is_checkmate():
                    total_wins += 1
                    game_end = True
                elif board.is_draw():
                    game_end = True
            board.reset()
            win_rate[epoch] = (total_wins / (epoch + 1)) * 100
        return win_rate

    def save(self, filename):
        """
        Function to save state space, avoiding re-computation.
        :param filename: the name file will be saved as
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.state_space, f, pickle.HIGHEST_PROTOCOL)
