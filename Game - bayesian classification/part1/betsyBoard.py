#!/usr/bin/env python3

from board import Board

class BetsyBoard( Board ):
    def __init__( self, n, stateString ):
        '''
        returns a board of size (n+3)xn to match the current state of the game
        for example for a board with n =3 and  current state given by:
            ...x..o.ox.oxxxoxo
        will be encoded as:
            [.,.,.]
            [X,.,.]
            [O,.,O]
            [X,.,O]
            [X,X,X]
            [O,X,O]
        '''
        if len( stateString ) != n * ( n + 3 ):
            return None

        Board.__init__( self, n + 3, n )

        # Storing value of move that resulted into this board
        # By default prevMove is 0
        # if you perform drop, this gets updated to <+(colNum)>
        # if you perform rotate, this gets updated to <-(colNum)>
        self.prevMove = 0

        currPos = 0
        for row in range(0,self.rowLen):
            self.board[row] = list( stateString[currPos:currPos+self.colLen] )
            currPos += self.colLen

    def getBoard(self):
        return self.board

    def getRowLen(self):
        return self.rowLen

    def getColLen(self):
        return self.colLen

    def getPrevMove(self):
        return self.prevMove

    def drop(self,col,pebble):
        '''
        # NOTE: The parameter passed assumes column number starts with 1

        return False if the column is filled, return True otherwise

        Choose the 'col' column, and drop the 'pebble' into that column.
        The pebble falls to occupy the bottom-most empty square in that column.
        The player is not allowed to choose a column that is already full (i.e.,
        already has n + 3 pebbles in it).
        '''
        prevRow = -1
        for row in range(0,self.rowLen):
            if self.board[row][col-1] != ".":
                row -= 1
                break

        if row < 0:
            # column is full
            return False
        else:
            self.board[row][col-1] = pebble

        # Update prevMove
        self.prevMove = col

        return True


    def rotate(self,col):
        '''
        # NOTE: The parameter passed assumes column number starts with 1

        return False if the column is empty, return True otherwise

        Choose the 'col' column, remove the pebble from the bottom of that column
        (whether red or blue) so that all pebbles fall down one square, and
        then drop that same pebble into the top of that column. The player is
        not allowed to choose an empty column for this type of move.
        '''

        bottomPebble = self.board[self.rowLen-1][col-1]

        # check if the column is empty
        if bottomPebble == ".":
            return False

        for row in range(self.rowLen-2,-1,-1):
            if self.board[row][col-1] == ".":
                break
            self.board[row+1][col-1] = self.board[row][col-1]

        self.board[row+1][col-1] = bottomPebble

        # Update prevMove
        self.prevMove = -1*col
        return True


    def getFlatString(self):
        return ("".join([ "".join([ self.board[r][c] \
            for c in range(0, self.colLen) ]) \
            for r in range(0, self.rowLen)]))

    def copy(self):
        '''
        Returns copy of self
        '''
        return BetsyBoard.create_from(self.board)
