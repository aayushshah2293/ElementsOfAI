#!/usr/bin/env python
# nrooks.py : Solve the N-Rooks problem!
# D. Crandall, 2016
# Updated by Zehua Zhang, 2017
#
# The N-rooks problem is: Given an empty NxN chessboard, place N rooks on the board so that no rooks
# can take any other, i.e. such that no two rooks share the same row or column.

import sys


# Count # of pieces in given row
def count_on_row(board, row):
    return sum(board[row])


# Count # of pieces in given column
def count_on_col(board, col):
    return sum([row[col] for row in board])


# Count total # of pieces on board
def count_pieces(board):
    return sum([sum(row) for row in board])


# Return a string with the board rendered in a human-friendly format
def printable_board(board):
    return "\n".join([" ".join(["R" if col == 1 else "X" if col == -1 else "_" for col in row]) for row in board])


# Return a string with the board rendered in a human-friendly format for n queen problem
def printable_board_queen(board):
    return "\n".join([" ".join(["Q" if col == 1 else "X" if col == -1 else "_" for col in row]) for row in board])

# Return a string with the board rendered in a human-friendly format for n knight problem
def printable_board_knight(board):
    return "\n".join([" ".join(["K" if col == 1 else "X" if col == -1 else "_" for col in row]) for row in board])


# Checks if a queen lies on diagonal of another queen.
def check_diagonals(board):
    list_of_queens = []
    check = 1
    for r in range(0, N):
        for c in range(0, N):
            if board[r][c] == 1:
                list_of_queens += [[r, c]]

    if len(list_of_queens) > 1:
        for Q1 in list_of_queens:
            for Q2 in list_of_queens:
                if Q1 != Q2:
                    diff_rows = abs(Q1[0] - Q2[0])
                    diff_cols = abs(Q1[1] - Q2[1])
                    if diff_rows == diff_cols:
                        check = 0
                        break
            if not check:
                break
    return check

# Checks if a Knight attacks another Knight.
def check_Knights(board):
    list_of_knights = []
    check = 1
    for r in range(0, N):
        for c in range(0, N):
            if board[r][c] == 1:
                list_of_knights += [[r, c]]

    if len(list_of_knights) > 1:
        for K1 in list_of_knights:
            for K2 in list_of_knights:
                if K1 != K2:
                    diff_rows = abs(K1[0] - K2[0])
                    diff_cols = abs(K1[1] - K2[1])
                    if (diff_rows == 2 and diff_cols == 1) or (diff_rows == 1 and diff_cols == 2):
                        check = 0
                        break
            if not check:
                break
    return check

# Add a piece to the board at the given position, and return a new board (doesn't change original)
def add_piece(board, row, col):
    return board[0:row] + [board[row][0:col] + [1, ] + board[row][col + 1:]] + board[row + 1:]

# Add a blocked square to the board at the given position, and return a new board (doesn't change original)
def add_piece_blank(board, row, col):
    return board[0:row] + [board[row][0:col] + [-1, ] + board[row][col + 1:]] + board[row + 1:]


# Get list of successors of given board state for nrook
def successors(board):
    result_successors = []
    column_count = [count_on_col(board, col) for col in range(0, N)]
    if 0 in column_count:
        c = column_count.index(0)
    else:
        return result_successors

    block = 0

    for r in range(0, N):
        if board[r][c] != 1 and count_pieces(board) <= N and count_on_row(board, r) != 1:
            if no_of_blanks > 0:
                for i in range(0, 2 * no_of_blanks, 2):
                    if r == int(blanks[i]) and c == int(blanks[i + 1]):
                        block = 1
                        break
            if block:
                block = 0
                continue
            result_successors += [add_piece(board, r, c)]

    return result_successors

# Get list of successors of given board state for nqueen
def successorsQ(board):
    temp1 = []
    result_successors = []
    column_count = [count_on_col(board, col) for col in range(0, N)]
    if(0 in column_count):
        c = column_count.index(0)
    else:
        return result_successors
    block = 0
    for r in range(0, N):
        if board[r][c] != 1 and count_pieces(board) <= N and count_on_row(board, r) != 1:
            if no_of_blanks > 0:
                for i in range(0, 2 * no_of_blanks, 2):
                    if r == int(blanks[i]) and c == int(blanks[i + 1]):
                        block = 1
                        break
            if block:
                block = 0
                continue
            temp1 = add_piece(board, r, c)
            if check_diagonals(temp1):
                result_successors += [temp1]

    return result_successors

def successorsK(board):
    temp1 = []
    result_successors = []
    block = 0

    for c in range(0, N):
        for r in range(0, N):
            if board[r][c] != 1 and count_pieces(board) <= N:
                if no_of_blanks > 0:
                    for i in range(0, 2 * no_of_blanks, 2):
                        if r == int(blanks[i]) and c == int(blanks[i + 1]):
                            block = 1
                            break
                if block:
                    block = 0
                    continue
                temp1 = add_piece(board, r, c)
                if check_Knights(temp1):
                    result_successors += [temp1]
    return result_successors

# check if board is a goal state
def is_goal(board):
    return count_pieces(board) == N and \
           all([count_on_row(board, r) <= 1 for r in range(0, N)]) and \
           all([count_on_col(board, c) <= 1 for c in range(0, N)])


# check if board is a goal state for nqueens
def is_goalQ(board):
    return count_pieces(board) == N and \
           all([count_on_row(board, r) <= 1 for r in range(0, N)]) and \
           all([count_on_col(board, c) <= 1 for c in range(0, N)]) and \
           check_diagonals(board)

# check if board is a goal state for nknights
def is_goalK(board):
    return count_pieces(board) == N and \
        check_Knights(board)

# Solve n-rooks!
def solve(initial_board):
    fringe = [initial_board]
    while len(fringe) > 0:
        # fringe.pop() makes fringe a stack hence DFS is used to find the solution
        # To convert from DFS to BFS we need to convert fringe from stackk to Queue by modifying fringe.pop() to fringe.pop(0)
        for s in successors(fringe.pop()):
            if is_goal(s):
                return (s)
            fringe.append(s)
    return False


# Solve n-queens!
def solve2(initial_board):
    fringe = [initial_board]
    while len(fringe) > 0:
        # fringe.pop() makes fringe a stack hence DFS is used to find the solution
        # To convert from DFS to BFS we need to convert fringe from stackk to Queue by modifying fringe.pop() to fringe.pop(0)
        for s in successorsQ(fringe.pop()):
            if is_goalQ(s):
                return (s)
            fringe.append(s)
    return False

# Solve n-knights!
def solve3(initial_board):
    fringe = [initial_board]
    while len(fringe) > 0:
        # fringe.pop() makes fringe a stack hence DFS is used to find the solution
        # To convert from DFS to BFS we need to convert fringe from stackk to Queue by modifying fringe.pop() to fringe.pop(0)
        for s in successorsK(fringe.pop()):
            if is_goalK(s):
                return (s)
            fringe.append(s)
    return False


# This is the problem type which decides if nrook or nqueen or nknight should be executed.
problem_type = sys.argv[1]

# This is N, the size of the board. It is passed through command line arguments.
N = int(sys.argv[2])

# This determines number of spots on board are blocked. It is passed through command line argument..
no_of_blanks = int(sys.argv[3])

# This array contains co-ordinates of the blocked spots. These are passed through commandline argument.
blanks = sys.argv[4:]

# converting co-ordinates from range 1,N to range 0,N-1
blanks = [int(x) - 1 for x in blanks]

# The board is stored as a list-of-lists. Each inner list is a row of the board.
# A zero in a given square indicates no piece, and a 1 indicates a piece.
initial_board = [[0] * N] * N

print("Starting from initial board:\n" + printable_board(initial_board) + "\n\nLooking for solution...\n")
if problem_type == "nrook":
    solution = solve(initial_board)
    if solution:
        if no_of_blanks > 0:
            for i in range(0, 2 * no_of_blanks, 2):
                r = int(blanks[i])
                c = int(blanks[i + 1])
                solution = add_piece_blank(solution, r, c)
        print(printable_board(solution))
    else:
        print("Sorry, no solution found. :(")
if problem_type == "nqueen":
    solution = solve2(initial_board)
    if solution:
        if no_of_blanks > 0:
            for i in range(0, 2 * no_of_blanks, 2):
                r = int(blanks[i])
                c = int(blanks[i + 1])
                solution = add_piece_blank(solution, r, c)
        print(printable_board_queen(solution))
    else:
        print("Sorry, no solution found. :(")
if problem_type == "nknight":
    solution = solve3(initial_board)
    if solution:
        if no_of_blanks > 0:
            for i in range(0, 2 * no_of_blanks, 2):
                r = int(blanks[i])
                c = int(blanks[i + 1])
                solution = add_piece_blank(solution, r, c)
        print(printable_board_knight(solution))
    else:
        print("Sorry, no solution found. :(")
