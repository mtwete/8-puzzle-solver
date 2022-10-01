"""
Matthew Twete

A simple 8 puzzle solver done with greedy and A-star breadth-first search.
Various heuristics were experimented with as well.
"""
import numpy as np
from queue import PriorityQueue
import math


#Goal state of the 8-puzzle given by the assignment
goalState = [1,2,3,4,5,6,7,8,0]
#Maximum number of steps the search is allowed to take
MAXSTEPS = 1000

#Function to check if the 8-puzzle can be solved from a given state
def solvable(state):
    count = 0
    #Count the number of inversions
    for i in range(9):
        for j in range(i + 1, 9):
            if (state[i] != 0 and state[j] < state[i] and state[j] != 0):
                count += 1
    #If the number of inversions is even, it is solvable, otherwise it is not
    if (count % 2 == 0):
        return True
    else:
        return False

    
#Check to see if the state is the goal state
def goalTest(state):
    listState = list(state)
    if (listState == goalState):
        return True
    else:
        return False

        
class BFS:
    def __init__(self): 
        #These initial states were randomly generated using np.random.shuffle 
        #I then hard coded them in so that the same ones would always be used
        self.initStates = [[4, 5, 3, 1, 0, 7, 2, 6, 8],
                           [0, 5, 7, 3, 1, 8, 2, 4, 6],
                           [2, 4, 5, 6, 3, 1, 0, 7, 8],
                           [2, 5, 1, 3, 8, 7, 6, 4, 0],
                           [8, 4, 6, 5, 0, 3, 2, 1, 7]]
        #Current state of search
        self.currentState = None
        #Array to hold number of steps taken for each of the three heuristics for the 5 initial state 
        self.steps = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
        #Array to hold paths taken for each of the three heuristics for the 5 initial state 
        self.paths = [[[],[],[],[],[],[]],[[],[],[],[],[],[]],[[],[],[],[],[],[]]]

    #Heuristic 1, number of misplaced tiles
    def h1(self, state):
        if (goalTest(state)):
            return 0
        cost = 0
        for i in range(9):
            if (state[i] != 0 and state[i] != goalState[i]):
                cost += 1
        return cost
    
    #Heuristic 2, manhatten distance
    def h2(self, state):
        if (goalTest(state)):
            return 0
        cost = 0
        goal = np.reshape(goalState, (3,3))
        st = np.reshape(state, (3,3))
        for row in st:
            for col in row:
                i, j = np.where(st == col)
                k, l = np.where(goal == col)
                if (col != 0):
                    cost += abs(int(k-i)) + abs(int(l-j))
        return cost
    
    #Heuristic 3, sum of eucledian distances of the tiles from their goal positions
    def h3(self, state):
        if (goalTest(state)):
            return 0
        cost = 0
        goal = np.reshape(goalState, (3,3))
        st = np.reshape(state, (3,3))
        for row in st:
            for col in row:
                i, j = np.where(st == col)
                k, l = np.where(goal == col)
                if (col != 0):
                    cost += math.sqrt(int(k-i)**2 + int(l-j)**2)
        return cost
    
    #Function to print the solution path
    def printSol(self,path):

        for j in range(5):
            #Check to see if the path list is empty (the initial state wasn't solvable)
            #if so, just print no solution and continue to next path
            if (len(path[j]) == 0 or goalTest(path[j][-1]) == False):
                print("No solution found")
                continue
            if (len(path[j]) <= 6):
                for i in range(len(path[j])):
                    if (i != len(path[j]) - 1):
                        print(path[j][i],"-> ",end='')
                    else:
                        print(path[j][i],flush = True)
            else:
                for i in range(3):
                    print(path[j][i],"-> ",end='')
                print("... -> ",end='')
                for i in range(len(path[j]) - 3, len(path[j])):
                    if (i != len(path[j]) - 1):
                        print(path[j][i],"-> ",end='')
                    else:
                        print(path[j][i],flush = True)
            print("")
                        
                        
    #Function to print the all 5 solution paths and the average number of steps
    def printResults(self):
        for i in range(3):
            print("Heuristic ", i+1,":")
            self.printSol(self.paths[i])
            print("Average number of steps: ", np.mean(self.steps[i]))
            print("")
            
            
    #Function to get the possible states you can go to from the current one,
    #to be used for the greedy search using f(n) = h(n)
    #the parameter h is passed in to tell the function which heuristic to use      
    def getPossibleStates1(self,state,h):
        #Reshape state into a 3x3 array and get the position of the empty tile
        state = np.reshape(state,(3,3))
        i, j = np.where(state == 0)
        i, j = int(i), int(j)
        #All possible positions you could move to
        posIndices = [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]
        for k in range(4):
            #Check to make sure that the new state is valid (i.e. it is not out of the bounds of the array)
            if(posIndices[k][0] >= 0 and posIndices[k][0] < 3 and posIndices[k][1] >= 0 and posIndices[k][1] < 3):
                #If the new state is valid, create it
                newState = np.copy(state)
                #Do the tile switch that leads to the new state
                newState[i,j], newState[posIndices[k]] = newState[posIndices[k]],newState[i,j]
                #Reshape and format the new state
                newState = list(np.reshape(newState,9))
                #For each of the three heuristics, if the new state isn't already in the closed list or 
                #in possibleStates, add it to the frontier
                if (h == 0):
                    if(newState not in self.closed and (self.h1(newState),newState) not in self.possibleStates.queue):
                        self.possibleStates.put((self.h1(newState),newState))
                elif (h == 1):
                    if(newState not in self.closed and (self.h2(newState),newState) not in self.possibleStates.queue):
                        self.possibleStates.put((self.h2(newState),newState))
                else:
                    if(newState not in self.closed and (self.h3(newState),newState) not in self.possibleStates.queue):
                        self.possibleStates.put((self.h3(newState),newState))
    
    #Function to get the possible states you can go to from the current one,
    #to be used for the A star search using f(n) = h(n) + g(n)
    #the parameter h is passed in to tell the function which heuristic to use 
    #the parameter cost is the cost to get up to the generated states
    def getPossibleStates2(self,state,h, cost):
        #Reshape state into a 3x3 array and get the position of the empty tile
        state = np.reshape(state,(3,3))
        i, j = np.where(state == 0)
        i, j = int(i), int(j)
        #All possible positions you could move to
        posIndices = [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]
        for k in range(4):
            #Check to make sure that the new state is valid (i.e. it is not out of the bounds of the array)
            if(posIndices[k][0] >= 0 and posIndices[k][0] < 3 and posIndices[k][1] >= 0 and posIndices[k][1] < 3):
                #If the new state is valid, create it
                newState = np.copy(state)
                #Do the tile switch that leads to the new state
                newState[i,j], newState[posIndices[k]] = newState[posIndices[k]],newState[i,j]
                #Reshape and format the new state
                newState = list(np.reshape(newState,9))
                #For each of the three heuristics, if the new state isn't already in the closed list or 
                #in possibleStates, add it to the frontier
                if (h == 0):
                    if((newState,cost) not in self.closed and (self.h1(newState) + cost,newState) not in self.possibleStates.queue):
                        self.possibleStates.put((self.h1(newState) + cost,newState))
                elif (h == 1):
                    if((newState,cost) not in self.closed and (self.h2(newState) + cost,newState) not in self.possibleStates.queue):
                        self.possibleStates.put((self.h2(newState) + cost,newState))
                else:
                    if((newState,cost) not in self.closed and (self.h3(newState) + cost,newState) not in self.possibleStates.queue):
                        self.possibleStates.put((self.h3(newState) + cost,newState))
                
            
    #Greedy BFS search
    def Greedy(self):
        #Loop to cover each of the three heuristics
        for i in range(3):
            #Loop to cover the five initial states
            for j in range(5):
                #To start set the current state to one of the inital states
                self.currentState = self.initStates[j]
                #First check if it is even solvable, if not, don't create a path list
                #and just set the number of steps to MAXSTEPS since the simulation would
                #just run until it reaches that number of steps
                if (solvable(self.currentState) == False):
                    self.steps[i][j] = MAXSTEPS
                    continue
                #If it is solvable, add the inital state to the path list and then start the
                #search
                self.paths[i][j].append(self.initStates[j])
                #Set up a priority queue to hold the states you could possibly move to
                self.possibleStates = PriorityQueue()
                #Set up list for states you have already visited
                self.closed = []
                #Loop to handle the steps of the search, goes up to MAXSTEPS
                for k in range(MAXSTEPS):
                    #If the current state is the goal state, stop
                    if(goalTest(self.currentState)):
                        break
                    #Expand the current state, adding the possible states you could move to the priority queue
                    self.getPossibleStates1(self.currentState,i)
                    #Add the current state to the closed list
                    self.closed.append(self.currentState)
                    #Get the next state which is of lowest estimated cost,
                    #possibleStates.get() gives you the lowest in the queue
                    nextState = self.possibleStates.get()
                    #possibleStates.get() returns a tuple with the cost & the state, so 
                    #get the actual state part of the tuple
                    nextState = nextState[1]
                    #Update the current state
                    self.currentState = nextState
                    #Add the new current state to the path
                    self.paths[i][j].append(nextState)
                    #Increment the number of steps taken by one
                    self.steps[i][j] += 1
    
    #A star BFS search                
    def aStar(self):
        #Loop to cover each of the three heuristics
        for i in range(3):
            #Loop to cover the five initial states
            for j in range(5):
                #To start set the current state to one of the inital states
                self.currentState = self.initStates[j]
                #First check if it is even solvable, if not, don't create a path list
                #and just set the number of steps to MAXSTEPS since the simulation would
                #just run until it reaches that number of steps
                if (solvable(self.currentState) == False):
                    self.steps[i][j] = MAXSTEPS
                    continue
                #If it is solvable, add the inital state to the path list and then start the
                #search
                self.paths[i][j].append(self.initStates[j])
                #Set up a priority queue to hold the states you could possibly move to
                self.possibleStates = PriorityQueue()
                #Set up list for states you have already visited
                self.closed = []
                #Loop to handle the steps of the search, goes up to MAXSTEPS
                for k in range(MAXSTEPS):
                    #If the current state is the goal state, stop
                    if(goalTest(self.currentState)):
                        break
                    #Expand the current state, adding the possible states you could move to the priority queue
                    self.getPossibleStates2(self.currentState,i,self.steps[i][j]+1)
                    #Add the current state and the number of steps it took to get there to the closed list
                    self.closed.append((self.currentState,self.steps[i][j]))
                    #Get the next state which is of lowest estimated cost,
                    #possibleStates.get() gives you the lowest in the queue
                    nextState = self.possibleStates.get()
                    #possibleStates.get() returns a tuple with the f(state) & the state
                    #so get the actual state part of the tuple
                    nextState = nextState[1]
                    #Update the current state
                    self.currentState = nextState
                    #Add the new current state to the path
                    self.paths[i][j].append(self.currentState)
                    #Increment the number of steps taken by one
                    self.steps[i][j] += 1
                    
                
                
                               
                
print("Greedy")
                
#Greedy best first search
g = BFS()
g.Greedy()
g.printResults()

print("A*")

#A star best first search
a = BFS()
a.aStar()
a.printResults()





                

