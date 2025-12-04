# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getAvailableActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateNextState(agentIndex, action):
        Returns the nextState game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minimax(state, agentIndex, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), None

            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            actions = state.getAvailableActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state), None

            bestAction = None

            if agentIndex == 0:  # Pacman (MAX)
                maxValue = float('-inf')
                for action in actions:
                    successor = state.generateNextState(agentIndex, action)
                    value, _ = minimax(successor, nextAgent, nextDepth)
                    if value > maxValue:
                        maxValue = value
                        bestAction = action
                return maxValue, bestAction
            else:  # Ghosts (MIN)
                minValue = float('inf')
                for action in actions:
                    successor = state.generateNextState(agentIndex, action)
                    value, _ = minimax(successor, nextAgent, nextDepth)
                    if value < minValue:
                        minValue = value
                        bestAction = action
                return minValue, bestAction

        _, bestAction = minimax(gameState, 0, 0)
        return bestAction

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphabeta(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), None

            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            actions = state.getAvailableActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state), None

            bestAction = None

            if agentIndex == 0:  # MAX - Pacman
                value = float('-inf')
                for action in actions:
                    successor = state.generateNextState(agentIndex, action)
                    successorValue, _ = alphabeta(successor, nextDepth, nextAgent, alpha, beta)
                    if successorValue > value:
                        value = successorValue
                        bestAction = action
                    if value > beta:  # Κλάδεμα (μόνο όταν > όχι >=)
                        break
                    alpha = max(alpha, value)
                return value, bestAction

            else:  # MIN - Ghost(s)
                value = float('inf')
                for action in actions:
                    successor = state.generateNextState(agentIndex, action)
                    successorValue, _ = alphabeta(successor, nextDepth, nextAgent, alpha, beta)
                    if successorValue < value:
                        value = successorValue
                        bestAction = action
                    if value < alpha:  # Κλάδεμα (μόνο όταν < όχι <=)
                        break
                    beta = min(beta, value)
                return value, bestAction

        _, action = alphabeta(gameState, 0, 0, float('-inf'), float('inf'))
        return action

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), None

            numAgents = state.getNumAgents()
            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            actions = state.getAvailableActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state), None

            if agentIndex == 0:  # Pacman (MAX)
                bestValue = float('-inf')
                bestAction = None
                for action in actions:
                    successor = state.generateNextState(agentIndex, action)
                    value, _ = expectimax(successor, nextDepth, nextAgent)
                    if value > bestValue:
                        bestValue = value
                        bestAction = action
                return bestValue, bestAction
            else:  # Ghosts (Chance Node: average value)
                totalValue = 0
                for action in actions:
                    successor = state.generateNextState(agentIndex, action)
                    value, _ = expectimax(successor, nextDepth, nextAgent)
                    totalValue += value
                averageValue = totalValue / len(actions)
                return averageValue, None

        _, bestAction = expectimax(gameState, 0, 0)
        return bestAction

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    foodList = food.asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    score = currentGameState.getScore()

    # --- Φαγητό
    if foodList:
        minFoodDist = min([manhattanDistance(pacmanPos, foodPos) for foodPos in foodList])
        score += 10.0 / (minFoodDist + 1)  # όσο πιο κοντά στο φαγητό, τόσο καλύτερα
        score += 100.0 / (len(foodList) + 1)  # όσο λιγότερο φαγητό μένει, τόσο καλύτερα
    else:
        score += 500  # όλα τα φαγητά φαγώθηκαν!

    # --- Κάψουλες (power pellets)
    score += 50 / (len(capsules) + 1)

    # --- Φαντάσματα
    for ghost in ghostStates:
        ghostDist = manhattanDistance(pacmanPos, ghost.getPosition())
        if ghost.scaredTimer > 0:
            # Αν είναι φοβισμένα, να τα κυνηγήσουμε
            score += 200.0 / (ghostDist + 1)
        else:
            if ghostDist < 2:
                score -= 500  # πολύ κοντά σε ενεργό φάντασμα = κίνδυνος
            else:
                score -= 5.0 / (ghostDist + 1)

    return score

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
