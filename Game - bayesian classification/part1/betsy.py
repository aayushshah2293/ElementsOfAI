#!/usr/bin/env python3
import time
import sys
import copy
from betsyBoard import BetsyBoard

'''
This code is used to play game of betsy. this program suggests the next move of the game given a state of the game
The rules of game is defined as:
It's played on a vertical board that is n squares wide and n + 3 squares tall
The board starts o empty, with each of the two players (red and blue) given
1/2n * (n + 3) pebbles of their own color. players can perform one of two possible types of moves.
• Drop: Choose one of the n columns, and drop a blue pebble into that column. The pebble falls to
occupy the bottom-most empty square in that column. The player is not allowed to choose a column
that is already full (i.e., already has n + 3 pebbles in it).
• Rotate: Choose one of the n columns, remove the pebble from the bottom of that column (whether
red or blue) so that all pebbles fall down one square, and then drop that same pebble into the top of
that column. The player is not allowed to choose an empty column for this type of move.

This program uses alpha-beta pruning algorithm to determine the best next move for the game 
and returns the move with the new board.

'''

def opponent_player(player):
    return 'o' if player == 'x' else 'x'

'''
This is the evaluation function which determines the likelihood of winning or loosing of any given board.
The function returns a score. Higher score represents player winning and lower score represents opponent winning.
The function is derived from evaluation function for tic-tac-toe given at https://kartikkukreja.wordpress.com/2013/03/30/heuristic-function-for-tic-tac-toe/

The evaluation function is divided into 3 parts. to evaluate score for each part there is a common lookup list created.
this list contains values corresponding to no. of player's pieces present or opponent's pieces present.
The list contains values increasing exponentially with final value defining winning or loosing condition with highest value
In current code we have selected exponent values of 5 since it gave a good balance in running time and sufficient distribution.

Heurestic is divided into 3 parts:
1. N*N Tic Tac Toe Matrix

The upper N*N matrix is similar to a game of Tic Tac Toe with similar winning conditions.
Here we calculate no. of player's piece and no. of opponent's pieces in a single row.
then we get score for no. of players pieces from lookup table and subtract the score for no. of opponent's pieces from lookup table
e.g. if in a particular row there are 3 'x' and 2 'o' and if the player is x then score for that row will be 
lookup[3]-lookup[2].
similarly score will be calculated for other rows, all columns and both diagonals.
and we all the scores together to get cumulative score

2. Lower N to N+3 rows of board

In this part of board your win is not counted. Hence we can ignore the row formations in this part. 
However a player can start filling in columns which can lead to a players win hence we calculate weights for each column
similar to calculation in 1st part and is added to previous cumulative score

3. Rotation of each columns

We find first complete row from top. Then we iteratively perform column rotation for each column in the row and calculate 
score for that row after each column rotation. This score is added to cumulative score.
This is performed for all complete rows from top till Nth row (leaving bottom three rows)

The final cumulative score is returned as a score for a particular state 
'''
def evaluation(boardObj,player):
    board = boardObj.getBoard()
    N = len(board[0])
    opponent = opponent_player(player)

    lookup = [0] + [5**k for k in range(1,N)] + [5**(2*N)]
    sum = 0
    playercountdia1 = 0
    opponentcountdia1 = 0
    playercountdia2 = 0
    opponentcountdia2 = 0
    #counting for rows,columns
    for i in range(0,N):
        playercountrow = 0
        opponentcountrow = 0
        playercountcol = 0
        opponentcountcol = 0
        for j in range (0,N):
            #counting no. of x and o in a row
            if board[i][j] == player:
                playercountrow += 1
            elif board[i][j] == opponent:
                opponentcountrow += 1

            # counting no. of x and o in a col
            if board[j][i] == player:
                playercountcol += 1
            elif board[j][i] == opponent:
                opponentcountcol += 1

            # counting no. of x and o in a dia1
            if i == j:
                if board[i][j] == player:
                    playercountdia1 += 1
                elif board[i][j] == opponent:
                    opponentcountdia1 += 1

            # counting no. of x and o in a dia2
            if i == N-1-j:
                if board[i][j] == player:
                    playercountdia2 += 1
                elif board[i][j] == opponent:
                    opponentcountdia2 += 1
        sum += (lookup[playercountrow] - lookup[opponentcountrow]) + (lookup[playercountcol] - lookup[opponentcountcol])

    sum += (lookup[playercountdia1] - lookup[opponentcountdia1]) + (lookup[playercountdia2] - lookup[opponentcountdia2])

    #considering bottom 3 rows of table
    for j in range(0, N):
        for i in range(N,N+3):
            lowerboardplayer = 0
            lowerboardopponent = 0
            if board[i][j] == player:
                lowerboardplayer += 1
            elif board[i][j] == opponent:
                lowerboardopponent += 1
        sum += (lookup[lowerboardplayer] - lookup[lowerboardopponent])


    #considering rotation of columns
    for i in range(0,N):
        if '.' in board[i]:
            continue
        for j in range (0,N):
            row = board[i].copy()
            pcount = 0
            ocount = 0
            if board[i-1][j] == '.' or i == 0:
                row[j] = board[N+3-1][j]
            else:
                row[j] = board[i-1][j]
            pcount = row.count(player)
            ocount = row.count(opponent)
            sum += lookup[pcount] - lookup[ocount]
    return sum

'''
This creates successors for a given board considering all possible moves which can be performed on the board.
It uses drop and rotate function from betsyBoard class to find all successors 
'''

def successors(boardObj,pebble):
    succList = []

    for c in range(1,boardObj.getColLen()+1):
        # for every column perform a drop and rotate
        temp = copy.deepcopy(boardObj)
        if temp.drop(c,pebble):
            succList.append(temp)

        temp = copy.deepcopy(boardObj)
        if temp.rotate(c):
            succList.append(temp)
    return succList

''''
It verifies if any given state is a winning state. i.e. if player or opponent has won
'''
def isgoal(board):
    N = len(board[0])
    dia1 = []
    dia2 = []
    for i in range(0,N):
        # check row
        row = board[i].copy()
        if all(row[j] == 'x' for j in range(0,N)) or all(row[j] == 'o' for j in range(0,N)):
            return True
        # check col
        col = [row[i] for row in board]
        if all(col[j] == 'x' for j in range(0, N)) or all(col[j] == 'o' for j in range(0, N)):
            return True
        # add dia1
        dia1 += board[i][i]
        # add dia2
        dia2 += board[i][N-1-i]
    # check dia1
    if all(dia1[i] == 'x' for i in range(0,N)) or all(dia1[i] == 'o' for i in range(0,N)):
        return True
    # check dia2
    if all(dia2[i] == 'x' for i in range(0,N)) or all(dia2[i] == 'o' for i in range(0,N)):
        return True
    return False


'''
represents the max node of game tree and also prunes the tree if the returned is lesser than current alpha
'''
def max_val(succ, alpha, beta, player, depth, h):

    if isgoal(succ.getBoard()) or depth == h:
        return evaluation(succ,player)
    for succ in successors(succ,player):
        alpha = max(alpha,min_val(succ, alpha, beta, player, depth + 1, h))
        if alpha >= beta:
            return alpha
    return alpha

'''
represents the min node of game tree and also prunes the tree if the returned is greater than current beta
'''
def min_val(succ, alpha, beta, player, depth, h):

    if isgoal(succ.getBoard()) or depth == h:
        return evaluation(succ,player)
    for succ in successors(succ, opponent_player(player)):
        beta = min(beta,max_val(succ, alpha, beta, player, depth + 1, h))
        if alpha >= beta:
            return beta
    return beta

'''
This is the main function of the program it initiates alpha-beta pruning for the game.
The depth of the tree is increased iteratively till the given cutoff time
It prints the best next move and new board for each depth
the depth is increased by 2 each time 
'''
def solve(boardObj,n,player,cutoff):
    # convert board to matrix using board string and n and store it in initial_board
    timeout = time.time() + cutoff
    h=1
    previousVal = -float("inf")
    while (time.time() < timeout):
        result = []
        nextMoveBoard = []
        for succ in successors(boardObj,player):
            val = min_val(succ,-float("inf"),float("inf"),player,1,h)
            result += [[val,succ]]
        (val, nextMoveBoard) = max(result,key=lambda x:x[0])
        if (val > previousVal):  # checks if current move is better than previous move
            print(nextMoveBoard.getPrevMove()," ",nextMoveBoard.getFlatString()) #print move,board
        previousVal = val
        h += 2

n = int(sys.argv[1]) #reads the size of board
boardState = sys.argv[3] #reads the current board state
player = sys.argv[2] #reads the player for which we are playing
cutoff = int(sys.argv[4]) #reads the time in seconds to perform move
#opponent = opponent_player(player)

'''
initialises the Board object with the current state of the board
'''
boardObj = BetsyBoard(n,boardState)
#print(boardObj)

# successors(boardObj,player)
# evaluation(boardObj,player)
solve(boardObj,n,player,cutoff)

# board = [['x', '.', '.'], ['o', '.', '.'], ['x', 'o', 'o'], ['o', 'o', 'o'], ['x', 'x', 'x'], ['x', 'x', 'o']]
# n=3
# player='x'
# print(evaluation(board,player))
# print(isgoal(board))
