import numpy as np
import abc
import util
from game import Agent, Action


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function take the average of all the possible score that can by done after the given action.
        """
        successor_game_state = current_game_state.generate_successor(action=action)

        avg_score = 0
        age_actions = successor_game_state.get_agent_legal_actions()
        for n_act in age_actions:
            avg_score += successor_game_state.generate_successor(agent_index=0, action=n_act).score
        if avg_score != 0:
            avg_score /= len(age_actions)

        return avg_score


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):

    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        max_s, a = self._max_score_action(game_state, self.depth)
        return a

    def _max_score_action(self, game_state, depth):
        '''
        Returns the max score action by using minmax search. each call of this function
        goes one level deeper in the tree by calculating the max score of the next minimal scores of all the successors.
        (recursively)
        :return: the best action with the best score of it.
        '''
        if depth == 0:
            return self.evaluation_function(game_state), Action.STOP

        max_score = 0
        age_next_actions = game_state.get_agent_legal_actions()
        if not age_next_actions: # no legal actions
            return self.evaluation_function(game_state), Action.STOP

        action = age_next_actions[0] # init value
        for act in age_next_actions:
            min_scores = []
            next_s = game_state.generate_successor(agent_index=0, action=act)
            for move in next_s.get_opponent_legal_actions():
                score, a = self._max_score_action(next_s.generate_successor(agent_index=1, action=move), depth-1)
                min_scores.append(score)
            min_s = min(min_scores)
            if min_s > max_score:
                max_score = min_s
                action = act

        return max_score, action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        max_s, a = self._max_score_action_p(game_state, self.depth)
        return a

    def _max_score_action_p(self, game_state, depth):
        '''
        Returns the max score action by using alphaBeta search. each call of this function
        goes one level deeper in the tree by calculating the max score of the next minimal scores of all the successors.
        (recursively).
        :return: the best action with the best score of it.
        '''
        if depth == 0:
            return self.evaluation_function(game_state), Action.STOP

        max_score = 0
        age_next_actions = game_state.get_agent_legal_actions()
        if not age_next_actions: # no legal actions
            return self.evaluation_function(game_state), Action.STOP

        action = age_next_actions[0] # init value
        for act in age_next_actions:
            min_scores = []
            next_s = game_state.generate_successor(agent_index=0, action=act)
            for move in next_s.get_opponent_legal_actions():
                score, a = self._max_score_action_p(next_s.generate_successor(agent_index=1, action=move), depth-1)
                min_scores.append(score)
                if score < max_score:
                    break
            min_s = min(min_scores)
            if min_s > max_score:
                max_score = min_s
                action = act

        return max_score, action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        max_s, a = self._max_score_action_e(game_state, self.depth)
        return a

    def _max_score_action_e(self, game_state, depth):
        '''
        Returns the max score action by using expectimax search. each call of this function
        goes one level deeper in the tree by calculating the max score of the next average scores of all the successors.
        (recursively).
        :return: the best action with the best score of it.
        '''
        if depth == 0:
            return self.evaluation_function(game_state), Action.STOP

        max_score = 0
        age_next_actions = game_state.get_agent_legal_actions()
        if not age_next_actions: # no legal actions
            return self.evaluation_function(game_state), Action.STOP

        action = age_next_actions[0] # init value
        for act in age_next_actions:
            min_scores = []
            next_s = game_state.generate_successor(agent_index=0, action=act)
            for move in next_s.get_opponent_legal_actions():
                score, a = self._max_score_action_e(next_s.generate_successor(agent_index=1, action=move), depth-1)
                min_scores.append(score)

            expectation_score = sum(min_scores)/len(min_scores)
            if expectation_score > max_score:
                max_score = expectation_score
                action = act

        return max_score, action


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).
    DESCRIPTION: The evaluation function based on 5 different elements of measuring the state.
    -Num of empty tile: as more empty tiles meaning to be far from stuck state.
    -Num of paris on board: The number of pairs of tiles that can be combined together in the
     next move = more scores+ more empty tiles.
    -Max tile vale.
    -Num of next legal actions = far from stuck state.
    -Score: the score of the game up to this state.

    The function returns the average of these values.
    """
    num_empty_t = len(current_game_state.get_empty_tiles())

    board = current_game_state.board
    h, w = board.shape
    num_pairs_v = 0
    for i in range(h):
        for j in range(w):
            if board[i, j] == 0:
                continue
            k = i + 1
            while (k < h):
                if board[i, j] == board[k, j]:
                    num_pairs_v += 1
                    break
                if board[k, j] == 0:
                    k += 1
                    continue
                else:
                    break
    num_pairs_h = 0
    for j in range(w):
        for i in range(h):
            if board[i, j] == 0:
                continue
            k = j + 1
            while (k < w):
                if board[i, j] == board[i, k]:
                    num_pairs_h += 1
                    break
                if board[i, k] == 0:
                    k += 1
                    continue
                else:
                    break
    a = 0.2
    b = 0.2
    c = 0.2
    d = 0.2
    e = 0.2
    return a * num_empty_t + b * max(num_pairs_h, num_pairs_v) + c * current_game_state.max_tile + \
           d * len(current_game_state.get_agent_legal_actions()) + e* current_game_state.score

# Abbreviation
better = better_evaluation_function
