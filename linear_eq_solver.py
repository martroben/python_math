# Write a function solve(*equations) that can solve systems of linear equations with 2, 3 to 6 variables. Equations will look like this:
# '2x+3y=6', '4x+9y=15'

# Return a dictionary with the results, i.e. {'x': 1.5, 'y': 1}. When solving higher order equation systems, variables can be named by any letter from a to z.

# You can assume that all equations are refined as much as possible so you don't have to handle cases like 2x+5x-2x-55y=2 but it will be in the form 5x-55y=2. However, quantifiers before variables might include floats so you have to handle cases like 0.5x+2y-4.6z=-1555.5. When dealing with floats, round the number to 2 decimal places so that the tester would accept your result. I.e. 2.55935312 -> 2.56.

# NB! numpy is not allowed.

# https://pyspace.eu/ws/thorgate/ch/23/



# https://www.programiz.com/python-programming/matrix


equations = [[2, -1, 4],
             [3, 2, 13]]


# Get multipliers of variables
def get_coefs(matrix):
    
    coef_matrix = []
    
    for row in matrix:
        coef_matrix.append(row[:-1])
    
    return coef_matrix
  
print(get_coefs(equations))


# Get constants
def get_consts(matrix):

    consts_vector = []
    
    for row in matrix:
        consts_vector.append(row[-1])
    
    return consts_vector
    
print(get_consts(equations))


# Get determinant of a 2x2 matrix
def get_2x2_det(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
    
print(get_2x2_det(get_coefs(equations)))


      
      
mat = [[1, 5, 3, 9], [2, 0, 4, 8], [6, 7, 5, 10], [0, -1, -2, -3]]
mat2 = [[2, 3], [4, 5]]


def get_submatrix(i, matrix):
    
    for k in range(1, len(matrix)):
        matrix[k].pop(i)
    
    matrix.pop(0)
    return matrix
    
def sign(col, row = 1):
    return (-1) ** (row + col)
    
    
print(sign(2))
