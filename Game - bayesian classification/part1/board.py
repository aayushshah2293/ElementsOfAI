#!/usr/bin/env python3

import random
import hashlib


class Board:

    def __init__(self, rowLen=1, colLen=1, missing_places=[]):
        '''
        Creates an empty board of size 'size'
        '''
        self.rowLen = rowLen
        self.colLen = colLen
        self.board = [['_'] * colLen for _ in range(rowLen)]
        for plc in missing_places:
            self.board[plc[0]][plc[1]] = 'X'

    @staticmethod
    def create_from(board_data):
        '''
        Creates a new board using given board config
        '''
        new_board = Board(len(board_data))
        new_board.board = [[cell for cell in row] for row in board_data]
        return new_board

    def copy(self):
        '''
        Returns copy of self
        '''
        return Board.create_from(self.board)

    def __str__(self):
        return '\n'.join([' '.join(map(str, row)) for row in self.board])

    def __hash__(self):
        return int(hashlib.sha256(''.join([''.join(map(str, row)) for row in self.board]).encode()).hexdigest(), 16)

    def __eq__(self, other):
        if not other:
            return False
        if self.rowLen != other.rowLen or self.colLen != other.colLen:
            return False
        return self.__hash__() == other.__hash__()
