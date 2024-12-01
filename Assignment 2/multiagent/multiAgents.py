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
import math #a necessary import.
import sys #a necessary import.
from pacman import SCARED_TIME #not quite necessary but nice to have this in evaluation function. 

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newGhostPositions = successorGameState.getGhostPositions() #retrieve the locations of the ghost. 
        numFood = successorGameState.getNumFood() #retrieve the number of foods left. 
        numGhosts = len(newGhostPositions) #compute the number of ghosts from their locations. 
        newCapsules = successorGameState.getCapsules() #retrieve the capsules. 
        allEatables = list(newFood.asList()) + newCapsules #combine foods and capsules as a list. 

        minGhostDistance = min(manhattanDistance(newPos, ghostPosition) for ghostPosition in newGhostPositions) #compute the minimum distance from pacman to any of the ghosts. 
        minEatableDistance = min(manhattanDistance(newPos, eatable) for eatable in allEatables) if allEatables else 0  #calculate the minimum distance from pacman to any of the foods or the capsules. 
        totalFoodDistance = sum(manhattanDistance(newPos, food) for food in newFood.asList()) #compute the total distance from pacman to all the foods. 

        if any(scaredTime > 0 for scaredTime in newScaredTimes): #if the pacman is in the chase mode. 
            if max(newScaredTimes) == SCARED_TIME: #if the pacman has just entered the chase mode. 
                return sys.maxsize #then, return the maximum value.
            return 1 / (minGhostDistance + 1) #else, the closer the ghost, the better the evaluation but we must bypass division by 0 so +1 to distance. 

        eval = (
            100000 / (numFood + 1) #prioritize eating the food, and avoid divison by 0. 
            + numGhosts / (minEatableDistance + 1) #consider eating the food that are closer, and avoid divison by 0. 
            + numGhosts / (totalFoodDistance + 1) #also consider other foods that are not necessarily close, and avoid division by 0.
            + 0.005 * (math.pow(numGhosts, 2)) * math.log(1 + minGhostDistance)) #consider minimum ghost distance, and avoid log 0. 
        eval = math.copysign(eval, math.log((1 + minGhostDistance) / 4)) #if the ghost is too near, then run. 
        
        return eval #return the final score. 

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.maxi(gameState, gameState.getNumAgents() * self.depth, 0)[0] #maxi determines the best move for pacman. 

    def minimax(self, gameState, depth, agent):
        if depth == 0 or gameState.isLose() or gameState.isWin(): #if we are at the terminal state, 
            return self.evaluationFunction(gameState) #then, just return the evaluation. 
        if agent == 0: #if it is pacman's turn, 
            return self.maxi(gameState, depth, agent)[1] #then, maximize. 
        else: #if it is ghosts' turn, 
            return self.mini(gameState, depth, agent)[1] #then, minimize. 

    def mini(self, gameState, depth, agent):
        minimum = ("action", float(sys.maxsize)) #initialize the minimal action with the maximum value for comparison. 
        for action in gameState.getLegalActions(agent): #for all possible actions, 
            possibleAction = (action, 
                              self.minimax(gameState.generateSuccessor(agent, action), 
                                           depth - 1, 
                                           (agent + 1) % gameState.getNumAgents())) #recursively compute the minimax value of the successor, and reduce the depth by 1. 
           
            if possibleAction[1] < minimum[1]: #if a possible action has a lesser value than the minimum, 
                minimum = possibleAction #then, update the minimal action accordingly. 
        
        return minimum #return the best found minimal action. 

    def maxi(self, gameState, depth, agent):
        maximum = ("action", -float(sys.maxsize)) #initialize the maximal action with the minimum value for comparison. 
        for action in gameState.getLegalActions(agent): #for all possible actions, 
            possibleAction = (action, 
                              self.minimax(gameState.generateSuccessor(agent, action), 
                                           depth - 1, 
                                           (agent + 1) % gameState.getNumAgents()))  #recursively compute the minimax value of the successor, and reduce the depth by 1. 
            
            if possibleAction[1] > maximum[1]: #if a possible action has a higher value than the minimum, 
                maximum = possibleAction  #then, update the maximal action accordingly. 
        
        return maximum #return the best found maximal action. 

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -float(sys.maxsize) #initialize the alpha value as negative infinity for comparison. 
        beta = float(sys.maxsize) #initialize the beta value as plus infinity for comparison. 
        return self.maxiAlphaBeta(gameState, (gameState.getNumAgents() * self.depth), alpha, beta, 0)[0] #maxiAlphaBeta determines the best move for pacman with alpha-beta pruning. 

    def alphaBeta(self, gameState, depth, alpha, beta, agent): 
        if depth == 0 or gameState.isLose() or gameState.isWin(): #if we are at the terminal state, 
            return self.evaluationFunction(gameState) #then, just return the evaluation. 
        if agent == 0:  #if it is pacman's turn, 
            return self.maxiAlphaBeta(gameState, depth, alpha, beta, agent)[1] #then, maximize. 
        else: #if it is ghosts' turn, 
            return self.miniAlphaBeta(gameState, depth, alpha, beta, agent)[1] #then, minimize. 

    def miniAlphaBeta(self, gameState, depth, alpha, beta, agent):
        minimum = ("action", float(sys.maxsize))  #initialize the minimal action with the maximum value for comparison. 
        for action in gameState.getLegalActions(agent): #for all possible actions, 
            possibleAction = (action, 
                              self.alphaBeta(gameState.generateSuccessor(agent, action), 
                                             depth - 1, 
                                             alpha, 
                                             beta, 
                                             (agent + 1) % gameState.getNumAgents())) #recursively compute the alphabeta value of the successor, and reduce the depth by 1. 
            
            if possibleAction[1] < minimum[1]: #if a possible action has a lesser value than the minimum, 
                minimum = possibleAction #then, update the minimal action accordingly. 
            
            if minimum[1] < alpha: #if the minimum value is less than alpha. 
                return minimum #then, prune the branch by returning minimal action immediately. 
            
            elif (beta > minimum[1]): #else if, the minimum is less than beta
                beta = minimum[1] #then, update beta according to the algorithm. 
        
        return minimum #return the minimal action at the end. 

    def maxiAlphaBeta(self, gameState, depth, alpha, beta, agent):
        maximum = ("action", -float(sys.maxsize))  #initialize the maximal action with the maximum value for comparison. 
        for action in gameState.getLegalActions(agent): #for all possible actions, 
            possibleAction = (action, 
                              self.alphaBeta(gameState.generateSuccessor(agent, action), 
                                             depth - 1, 
                                             alpha, 
                                             beta, 
                                             (agent + 1) % gameState.getNumAgents())) #recursively compute the alphabeta value of the successor, and reduce the depth by 1. 
            
            if possibleAction[1] > maximum[1]: #if a possible action has a higher value than the maximum, 
                maximum = possibleAction #then, update the maximal action accordingly. 
            
            if maximum[1] > beta: #if the maximum value is greater than beta. 
                return maximum #then, prune the branch by returning maximal action immediately. 
            
            elif (alpha < maximum[1]): #else if, the maximum is greater than alpha
                alpha = maximum[1] #then, update alpha according to the algorithm. 
        
        return maximum #return the maximal action at the end. 


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectiMax(gameState, (gameState.getNumAgents()*self.depth), "action", 0)[0] #call expectiMax for pacman. 
    
    def expectiMax(self, gameState, depth, action, agent):
        
        if depth == 0 or gameState.isLose() or gameState.isWin(): #if we are at the terminal state, 
            return (action, float(self.evaluationFunction(gameState))) #then, just return the current action with the current evaluation. 
            
        if agent == 0: #if its pacman's turn, 
            maximum = ("action", -(float(sys.maxsize))) #initialize the maximal action with the maximum value for comparison. 
            for pacmanAction in gameState.getLegalActions(agent): #for all possible actions, 
                if depth == (gameState.getNumAgents()*self.depth): #if we are at the root depth, 
                    possibleAction = pacmanAction #then, use the current action as the possible action. 
                else: #if not,
                    possibleAction = action #then, set the possible action as the action that has been passed. 
                
                probableAction = self.expectiMax(gameState.generateSuccessor(agent, pacmanAction), 
                                                 depth - 1, 
                                                 possibleAction, 
                                                 (agent + 1)%gameState.getNumAgents()) #recursively calculate the utility of the successor state, and reduce the depth by 1. 
                
                if probableAction[1] > maximum[1]: #if any of them are better than the current maximum, 
                    maximum = (possibleAction, probableAction[1]) #then, update the maximum accordingly. 

            return maximum #return the maximal action. 
        
        else: #if its ghosts' turn
            ghostEval = 0.0 #initialize expected utility of ghost as 0. 
            for ghostAction in gameState.getLegalActions(agent): #for all possible actions, 
                possibleAction = self.expectiMax(gameState.generateSuccessor(agent, ghostAction), 
                                                 depth - 1, 
                                                 action, 
                                                 (agent + 1)%gameState.getNumAgents()) #recursively calculate the utility of the successor state, and reduce the depth by 1. 
                
                ghostEval += (possibleAction[1] * 1.0/len(gameState.getLegalActions(agent))) #update the expected utility by averaging over all ghost actions.
            
            return (action, ghostEval) #return the action with the calculated expected utility. 
            

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPosition = currentGameState.getPacmanPosition() #retrieve current position of the pacman.
    currentFoods = currentGameState.getFood() #retrieve the current positions of the foods. 
    currentGhostStates = currentGameState.getGhostStates() #retrieve the current states of the ghosts. 
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates] #compute current scared times of the ghosts. 
    currentCapsules = currentGameState.getCapsules() #retrieve the positions of the capsules. 

    ghostPositions = currentGameState.getGhostPositions() #retrieve the current locations of the ghost.
    numFood = currentGameState.getNumFood() #retrieve the current number of foods left. 
    numGhosts = len(ghostPositions) #compute the current number of ghosts.
    numCapsules = len(currentCapsules) #compute the current number of capsules. 

    minFoodDistance = min(manhattanDistance(pacmanPosition, food) for food in currentFoods.asList()) if currentFoods.asList() else 0  #calculate the minimum distance to all foods and capsule. 
    maxFoodDistance = max(manhattanDistance(pacmanPosition, food) for food in currentFoods.asList()) if currentFoods.asList() else 0  #calculate the maximmum distance to all foods and capsule.
    minGhostDistance = min(manhattanDistance(pacmanPosition, ghostPosition) for ghostPosition in ghostPositions) #compute the minimum distance from pacman to any of the ghosts. 

    if any(scaredTime > 0 for scaredTime in currentScaredTimes): #if the pacman is in the chase mode. 
        return sys.maxsize / (minGhostDistance + 1) #prioritize minimizing the distance to the nearest ghost, and avoid division by 0. 
        
    eval = (
        10 * (numGhosts / (maxFoodDistance + 1)) #consider going towards clustered food, and avoid division by 0. 
        + 1000 * (numGhosts / (minFoodDistance + 1)) #consider going towards closer food, and avoid division by 0. 
        + 10000 / (numCapsules + 1) #promote eating capsules, and avoid division by 0. 
        + 10000000 / (numFood + 1)  #prioritize the remaining food, and avoid division by 0. 
        + 0.0045 * (math.pow(numGhosts, 2) * math.log(1 + minGhostDistance))) #consider minimum ghost distance, and avoid log 0. 
    eval = math.copysign(eval,  math.log((minGhostDistance + 1) / 2)) #if the ghost is too near, then run. 

    return eval #return the final evaluation. 
       

# Abbreviation
better = betterEvaluationFunction
