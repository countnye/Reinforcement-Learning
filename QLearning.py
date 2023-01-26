import random as r

import chess

import StateSpace as stateSpace
import ChessBoard as cb


class QLearning:
    """
    Q-Learning class
    """

    def __init__(self, epochs, alpha, gamma, epsilon, chess_board):
        self.epochs = epochs
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.e_decay = 0.9
        self.chess_board = chess_board
        self.state_space_model = stateSpace.StateSpace(cb.Scenario.KKR)
        self.state_space = self.state_space_model.load('state_spaceKKR.pkl')
        self.end_loop = False
        self.new_board = chess_board
        self.stats = {'wins': 0,
                      'game_length': 0,
                      'games_played': 0}

    def learn(self):
        """
        Function to implement Q-Learning.
        Right now function is using old moves
        as keys with new states for some reason
        """
        wins = 0
        game_length = 0
        for epoch in range(self.epochs + 1):
            epsilon_store = self.epsilon
            if epoch % 10 == 0:
                self.record_stats(wins, game_length, epoch)
            while not self.end_loop:
                curr_state = self.chess_board.get_board_representation()
                if r.uniform(0, 1) < self.epsilon:
                    move = self.chess_board.get_random_move()
                else:
                    _, move = self.get_next(self.chess_board)
                # get Q(s_t,a_t)
                curr_val = self.get_q_val(curr_state, move)
                # get max Q(s_{t+1}, a)
                next_state_board = self.chess_board.copy_board(curr_state, self.chess_board.board.turn)
                next_state_board.make_move(move)
                next_state_board.switch_turns()
                # if the move results in checkmate, reward is 100
                if next_state_board.is_checkmate():
                    reward = 100
                # if the move results in stalemate, reward is 0
                elif next_state_board.is_draw():
                    reward = -100
                # IMPLEMENT A GOOD HEURISTIC
                else:
                    reward = 7 - self.chess_board.get_king_distance() - self.chess_board.get_corner_distance()
                # if the next state is a stalemate, Q-value should decrease
                next_q_val = 0
                # else get max Q-value
                if len(next_state_board.get_legal_moves()) != 0:
                    next_q_val, _ = self.get_next(next_state_board)
                game_length = self.chess_board.board.fullmove_number
                # calculate TD error
                td_error = reward + self.gamma * next_q_val - curr_val
                # update Q value
                new_q_val = curr_val + self.alpha * td_error
                self.state_space[curr_state][move] = new_q_val
                # make the move
                self.chess_board.make_move(move)
                self.epsilon *= self.e_decay
                # end loop if checkmate
                if self.chess_board.is_checkmate():
                    self.end_loop = True
                    wins += 1
                elif self.chess_board.is_draw():
                    self.end_loop = True
            # reset the board after each epoch
            self.chess_board.reset()
            self.end_loop = False
            self.epsilon = epsilon_store

    def get_q_val(self, state, action):
        """
        Function to get the Q-Value of the given state.
        :param state: the representation of current state
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
            if curr_q_val > q_val:
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

    def get_stats(self):
        """
        Function to get the stats of the game.
        Stats are given by (epoch_num, wins, game length, games played).
        :return: the stats
        """
        return self.stats

    def record_stats(self, wins, game_length, games_played):
        """
        Function to record the stats of the game.
        :param wins: the number of wins
        :param game_length: the number of moves made
        :param games_played: the number of games played
        """
        self.stats['wins'] = wins
        self.stats['game_length'] = game_length
        self.stats['games_played'] = games_played
