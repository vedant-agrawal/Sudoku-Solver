# This file contains the solve function that is used to solve the board once the numbers are predicted
# It uses the Backtracking Algorithm

def isEmpty(board):
    
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                return (i, j) 
    
    return None

def isValid(board, num, pos):
    
    for i in range(len(board[0])):
        
        if board[pos[0]][i] == num and pos[1] != i:
            return False
        if board[i][pos[1]] == num and pos[0] != i:
            return False

    x = pos[1] // 3
    y = pos[0] // 3
    for i in range(y*3, y*3 + 3):
        for j in range(x * 3, x*3 + 3):
            if board[i][j] == num and (i,j) != pos:
                return False
    
    return True

def solve(board):
    empty = isEmpty(board)
    if not empty:
        return True

    row, col = empty
    for i in range(1,10):
        if isValid(board, i, empty):
            board[row][col] = i
            if solve(board):
                return True
            board[row][col] = 0
    return False