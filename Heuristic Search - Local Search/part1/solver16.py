#!/usr/bin/env python3
# solver16.py : Circular 16 Puzzle solver
# Based on skeleton code by D. Crandall, September 2018
#

'''
Formulation: The 16 puzzle problem can be considered as a search problem in which we try to find a set of moves
of sliding rows or columns that converts the board into goal state

Abstraction:

Initial State: A scrambled 4x4 board containing numbers 1-16 in random order.

Goal State: A 4x4 board with tiles arranged in ascending order from 1-16.

State Space: All possible combinations of scrambled board. All the state objects are stored in fringe(priority queue)

Successor Function: This function generates successor for a given state by sliding each row in left and right direction and
sliding each column in up and down direction. in total this function will generate 16 different states from a given state.

Cost: Since it takes 1 move to go from a given state to its successor we are considering 1 as uniform cost.

Heuristics Function:
We have tried following Heuristics Functions:

1. Manhattan Distance (separating row and column movements) Currently being used code in h6:
This Heuristic is derived from a heuristic suggestion on Piazza in post @263 https://piazza.com/class/jl1erlsbz1n6ax?cid=263# suggested by Xiaomeng Ye

The Heuristic is a variation of Manhattan Distance however  it considers row and column movement separately.
It calculates the how much displaced each element in a particular row is from its actual column in left direction and right direction separately.
then for each direction maximum of these displacements is picked in a row to consider most displaced tile in a row and then minimum of both values is picked.
this is done for each row.

Similarly it calculates how much displaced each element is from its actual row.
These displacements are then added.
However either row or column move will be done on the state to progress to next state we divide this sum by 2.
This is a admissible and consistent heuristic. hence we can discard visited states

This heuristic provides optimal results till board with 11 moves within a minute. However it is not able to calculate results for board 12 where it runs out of memory

2. Manhattan Distance (calculating Manhattan Distance considering wrap around condition) code in h2:

In this heuristic we calculate the Manhattan Distance but not in conventional way. since we can wrap around rows and columns and both can be moved in both directions
during calculations whenever we find the difference in x co-ordinates or y co-ordinates is greater than or equal to 3 we consider it as one since instead of moving 3 moves in that direction
we can perform 1 move in opposite direction to have same effect. we calculate such Manhattan Distance for each element and add them.

However whenever we perform a move we dont move a single tile we affect 4 tiles in a single move.
Hence we divide this Manhattan distance by 4.

This is admissible heuristic however it fails consistency test. Hence while implementation of this heuristic we do not discard visited boards.

This beuristic also gives optimal results till board 11 but is slower than heuristic 1

3. No. of Misplaced Tiles code in h1

This heuristic considers no. of misplaced tiles on the board. it takes sum of number of misplaced tiles on the edge of
the board and in center of the board separately and divides the misplaced tiles on edge by 4 and no of misplaced tiles in center by 2.

this is also a admissible heuristic however it is not consistent as well as has very low efficiency

Evaluation Function:
this determines the priority of the successor in the priority queue.
it is calculated as sum of heuristic and cost functions
E(s) = H(s) + g(s)

The code currently uses Heuristic function 1 which is present in function h6()
'''
from queue import PriorityQueue
from random import randrange, sample
import sys
import string
import numpy as np
import time
from operator import sub

# shift a specified row left (1) or right (-1)
def shift_row(state, row, dir,no_of_moves):
    change_row = state[(row*4):(row*4+4)]
    temp = state[:(row*4)] + change_row[-dir:] + change_row[:-dir] + state[(row*4+4):]
    return ((h6(temp) + no_of_moves + 1), temp , ("L" if dir == -1 else "R") + str(row+1), (no_of_moves + 1))

# shift a specified col up (1) or down (-1)
def shift_col(state, col, dir, no_of_moves):
    change_col = state[col::4]
    s = list(state)
    s[col::4] = change_col[-dir:] + change_col[:-dir]

    return ((h6(s) + no_of_moves + 1),s ,("U" if dir == -1 else "D") + str(col+1) , (no_of_moves + 1))

# pretty-print board state
def print_board(row):
    for j in range(0, 16, 4):
        print ('%3d %3d %3d %3d' % (row[j:(j+4)]))

# return a list of possible successor states
def successors(state, no_of_moves):
    res = []
    for i in range(0,4):
        for d in (1,-1):
            res += [shift_row(state,i,d,no_of_moves)]
    for i in range(0,4):
        for d in (1,-1):
            res += [shift_col(state,i,d,no_of_moves)]
    return res

# Heuristic 3 : No. of Misplaced Tiles
def h1(board):
    goal = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    board_arr = np.array(board).reshape(4, 4)
    goal_arr = np.array(goal).reshape(4, 4)

    edge, non_edge = 0, 0
    for i in range(4):
        for j in range(4):
            if board_arr[i][j] != goal_arr[i][j]:
                if i == 0 or i == 3 or j == 0 or j == 3:
                    edge += 1
                else:
                    non_edge += 1
    return edge // 4 + non_edge // 2

# Heuristic 2 : Manhattan Distance (calculating Manhattan Distance considering wrap around condition)
def h2(board):
    board_arr = np.array(board).reshape(4, 4)
    manhattan_distance = [ ([0] * 4) for row in range(4) ]
    for i in range(4):
        for j in range(4):
            manhattan_distance[i][j] = int(Get_Manhattan_Distance(board_arr[i][j],i,j))

    return (sum(map(sum, manhattan_distance))/4)


# Heuristic 1 : Manhattan Distance (separating row and column movements)
def h6(board):
    board_arr = np.array(board).reshape(4, 4)
    manhattan_distance_LD = [([0] * 4) for row in range(4)]
    manhattan_distance_RU = [([0] * 4) for row in range(4)]
    left = []
    down = []
    up = []
    right = []
    for i in range(4):
        for j in range(4):
            manhattan_distance_LD[i][j] = Get_Manhattan_Distance_LD(board_arr[i][j], i, j)

    for i in range(4):
        for j in range(4):
            manhattan_distance_RU[i][j] = Get_Manhattan_Distance_RU(board_arr[i][j], i, j)

    for i in range(4):
        left += [list(manhattan_distance_LD[i][col][1] for col in range(4))]

    for i in range(4):
        down += [list(manhattan_distance_LD[row][i][0] for row in range(4))]

    for i in range(4):
        right += [list(manhattan_distance_RU[i][col][1] for col in range(4))]

    for i in range(4):
        up += [list(manhattan_distance_RU[row][i][0] for row in range(4))]
    sum = 0
    for i in range(4):
        temp1 = min(max(left[i]),max(right[i]))
        temp2 = min(max(down[i]),max(up[i]))
        temp3 = temp1+temp2
        sum += temp3
    return sum/2

# calculates Manhattan Distance of given element considering wrap around condition
def Get_Manhattan_Distance(value,x,y):
    goal = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    goal_arr = np.array(goal).reshape(4, 4)
    gX,gY = np.where(goal_arr == value)

    xDiff = abs(x - gX)
    yDiff = abs(y - gY)

    if xDiff >=3:
        xDiff = 1

    if yDiff >=3:
        yDiff = 1

    return (xDiff + yDiff)

# calculate Manhattan Distance of given element in left and down direction and return list of x and y coordinate
def Get_Manhattan_Distance_LD(value,x,y):
    goal = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    goal_arr = np.array(goal).reshape(4, 4)
    gX,gY = np.where(goal_arr == value)

    xDiff = gX - x
    yDiff = gY - y

    if xDiff < 0 :
        xDiff = 4+xDiff

    if yDiff < 0:
        yDiff = 4+yDiff


    return list([int(xDiff), int(yDiff)])

# calculate Manhattan Distance of given element in right and up direction and return list of x and y coordinate
def Get_Manhattan_Distance_RU(value,x,y):
    goal = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    goal_arr = np.array(goal).reshape(4, 4)
    gX,gY = np.where(goal_arr == value)

    xDiff = x - gX
    yDiff = y - gY

    if xDiff < 0 :
        xDiff = 4+xDiff

    # if xDiff <=-3:
    #     xDiff = 1

    if yDiff < 0:
        yDiff = 4+yDiff

    # if yDiff <=-3:
    #     yDiff = 1
    #temp = [int(xDiff), int(yDiff)]
    return list([int(xDiff), int(yDiff)])

# just reverse the direction of a move name, i.e. U3 -> D3
def reverse_move(state):
    return state.translate(string.maketrans("UDLR", "DURL"))

# check if we've reached the goal
def is_goal(state):
    return sorted(state) == list(state)
    
# The solver! - using BFS right now
def solve(initial_board):
    fringe = PriorityQueue()
    previous_board = []
    visited = []
    successorNo = 1
    fringe.put((0,0, [ (initial_board, "", 0) ]))
    while not fringe.empty():
        (priority, succno ,[(state, route_so_far, no_of_moves)]) = fringe.get()
        for (pri,succ, move, no_of_moves) in successors( state, no_of_moves ):

            if is_goal(succ):
                return( route_so_far + " " + move )
            if succ not in visited:
                fringe.put((pri,successorNo, [(succ, route_so_far + " " + move, no_of_moves )] ))
                successorNo+=1
                visited +=[succ]
        # below section should be uncommented if Heuristic 2 is being used
        #     if(succ!= previous_board):
        #         fringe.put((pri, successorNo, [(succ, route_so_far + " " + move, no_of_moves)]))
        #         successorNo += 1
        # previous_board = initial_board
    return False

# test cases
start_time = time.time()
start_state = []
with open(sys.argv[1], 'r') as file:
    for line in file:
        start_state += [ int(i) for i in line.split() ]

if len(start_state) != 16:
    print ("Error: couldn't parse start state file")

print ("Start state: ")
print_board(tuple(start_state))

print ("Solving...")
route = solve(start_state)

print("--- %s seconds ---" % (time.time() - start_time))

print ("Solution found in " + str(len(route)/3) + " moves:" + "\n" + route)





