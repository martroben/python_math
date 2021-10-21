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


mat = [[1, 5, 3, 9],
       [2, 0, 4, 8],
       [6, 7, 5, 10],
       [0, -1, -2, -3]]

mat2 = [[2, 3],
        [4, 5]]

mat3 =[[1, 3, 5, 9],
       [1, 3, 1, 7],
       [4, 3, 9, 7],
       [5, 2, 0, 9]]


# i - rows, j - cols
def get_submatrix(matrix, i, j):
    
    submatrix = []
    for k in range(len(matrix)):
        submatrix.append(matrix[k][:(j-1)] + matrix[k][j:])
    
    submatrix.pop(i-1)

    return submatrix
    
    
def cofactor_sign(i, j):
    return (-1) ** (i + j)


def get_determinant(matrix):
    
    if len(matrix) == 1:
        return matrix[0][0]

    determinant = 0
    for n in range(len(matrix)):
        submatrix = get_submatrix(matrix, i = 1, j = n + 1)
        cofactor = cofactor_sign(i = 1, j = n + 1) * get_determinant(submatrix)
        determinant += matrix[0][n] * cofactor

    return determinant

print(get_determinant(mat3))


# not invertible if determinant = 0
