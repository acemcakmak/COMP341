# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    "*** YOUR CODE HERE ***"
    frontier, paths = util.Stack(), util.Stack() #Initialize frontier and paths as Stack (LIFO) data structure. 
    visited = [] #Initialize a list of visited states to bypass revisiting. 
    frontier.push(problem.getStartState()) #Push the initial state into frontier to start exploration. 
    paths.push([]) #Push the initial path (an empty list) into paths to start exploration.

    while not frontier.isEmpty(): #If frontier is not empty, then we still have nodes to explore.
        current_state = frontier.pop() #Initialize current state by popping the last elemeent of frontier. 
        current_path = paths.pop() #Initialize current path by popping the last elemeent of paths. 

        if current_state not in visited: #Check if current state has been visited or not.
            visited.append(current_state) #If not, then add current state to the visited states list. 
            
            if problem.isGoalState(current_state): #Check if current state is the goal state or not. 
                return current_path #If so, then just return the current path. 
            successors = problem.getSuccessors(current_state) #If not, then start exploring the successors. 
            
            for successor in successors: #For each successor in the successors.
                if successor[0] not in visited: #Check if that node has been visited or not. 
                    frontier.push(successor[0]) #If not, then push that successor into frontier. 
                    paths.push(current_path + [successor[1]]) #As well as the current path appended by the path to that successor into paths. 
    
    return [] #Goal state could not be achieved, return an empty list. 

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""

    "*** YOUR CODE HERE ***"
    frontier, paths = util.Queue(), util.Queue() #Initialize frontier and paths as Queue (FIFO) data structure. 
    visited = [] #Initialize a list of visited states to bypass revisiting. 
    frontier.push(problem.getStartState()) #Push the initial state into frontier to start exploration. 
    paths.push([]) #Push the initial path (an empty list) into paths to start exploration.

    while not frontier.isEmpty(): #If frontier is not empty, then we still have nodes to explore.
        current_state = frontier.pop() #Initialize current state by popping the last elemeent of frontier. 
        current_path = paths.pop() #Initialize current path by popping the last elemeent of paths. 

        if current_state not in visited: #Check if current state has been visited or not.
            visited.append(current_state) #If not, then add current state to the visited states list. 
            
            if problem.isGoalState(current_state): #Check if current state is the goal state or not. 
                return current_path #If so, then just return the current path. 
            successors = problem.getSuccessors(current_state) #If not, then start exploring the successors. 
            
            for successor in successors: #For each successor in the successors.
                if successor[0] not in visited: #Check if that node has been visited or not. 
                    frontier.push(successor[0]) #If not, then push that successor into frontier. 
                    paths.push(current_path + [successor[1]]) #As well as the current path appended by the path to that successor into paths. 
    
    return [] #Goal state could not be achieved, return an empty list. 

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""

    "*** YOUR CODE HERE ***"
    frontier, paths = util.PriorityQueue(), util.PriorityQueue() #Initialize frontier and paths as PriorityQueue data structure, so that the least cost node is explored first. 
    visited = [] #Initialize a list of visited states to bypass revisiting. 
    frontier.push(problem.getStartState(), 0) #Push the initial state with 0 cost into frontier to start exploration. 
    paths.push([], 0) #Push the initial path (an empty list) with 0 cost into paths to start exploration.

    while not frontier.isEmpty(): #If frontier is not empty, then we still have nodes to explore.
        current_state = frontier.pop() #Initialize current state by popping the last elemeent of frontier. 
        current_path = paths.pop() #Initialize current path by popping the last elemeent of paths. 
        
        if current_state not in visited: #Check if current state has been visited or not.
            visited.append(current_state) #If not, then add current state to the visited states list. 

            if problem.isGoalState(current_state): #Check if current state is the goal state or not. 
                return current_path #If so, then just return the current path. 
            successors = problem.getSuccessors(current_state) #If not, then start exploring the successors. 
            
            for successor in successors: #For each successor in the successors.
                if successor[0] not in visited: #Check if that node has been visited or not. 
                    frontier.push(successor[0], problem.getCostOfActions(current_path + [successor[1]])) #If not, then push that successor into frontier with the associated cost to that successor. 
                    paths.push(current_path + [successor[1]], problem.getCostOfActions(current_path + [successor[1]])) #As well as the current path appended by the path to that successor with the associated cost to that successor into paths. 

    return []  #Goal state could not be achieved, return an empty list. 

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    
    "*** YOUR CODE HERE ***"
    frontier, paths = util.PriorityQueue(), util.PriorityQueue() #Initialize frontier and paths as PriorityQueue data structure, so that the least cost node is explored first. 
    visited = [] #Initialize a list of visited states to bypass revisiting. 
    frontier.push(problem.getStartState(), 0) #Push the initial state with 0 cost into frontier to start exploration. 
    paths.push([], 0) #Push the initial path (an empty list) with 0 cost into paths to start exploration.

    while not frontier.isEmpty(): #If frontier is not empty, then we still have nodes to explore.
        current_state = frontier.pop() #Initialize current state by popping the last elemeent of frontier. 
        current_path = paths.pop() #Initialize current path by popping the last elemeent of paths. 
        
        if current_state not in visited: #Check if current state has been visited or not.
            visited.append(current_state) #If not, then add current state to the visited states list. 

            if problem.isGoalState(current_state): #Check if current state is the goal state or not. 
                return current_path #If so, then just return the current path. 
            successors = problem.getSuccessors(current_state) #If not, then start exploring the successors. 
            
            for successor in successors: #For each successor in the successors.
                if successor[0] not in visited: #Check if that node has been visited or not. 
                    frontier.push(successor[0], heuristic(successor[0], problem) + problem.getCostOfActions(current_path + [successor[1]])) #If not, then push that successor into frontier with the associated cost to that successor. 
                    paths.push(current_path + [successor[1]], heuristic(successor[0], problem) + problem.getCostOfActions(current_path + [successor[1]])) #As well as the current path appended by the path to that successor with the associated cost to that successor into paths. 

    return []  #Goal state could not be achieved, return an empty list. 
    

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
