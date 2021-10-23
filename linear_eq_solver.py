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



def get_cofactor(matrix, i, j):
    
    submatrix = get_submatrix(matrix, i, j)
    cofactor_sign = (-1) ** (i + j)
    cofactor = cofactor_sign * get_determinant(submatrix)
    
    return cofactor


def get_determinant(matrix):
    
    if len(matrix) == 1:
        return matrix[0][0]

    determinant = 0
    for n in range(len(matrix)):
        determinant += matrix[0][n] * get_cofactor(matrix, i = 1, j = n + 1)

    return determinant
 

def get_transpose_matrix(matrix):

    matrix_T = []
    n_vec = range(len(matrix))
    for i in n_vec:
        row_T = [matrix[j][i] for j in n_vec]
        matrix_T.append(row_T)

    return matrix_T


def scalar_x_matrix(scalar, matrix):
    
    product = []
    n_vec = range(len(matrix))
    for i in n_vec:
        product_row = [scalar * matrix[i][j] for j in n_vec]
        product.append(product_row)
    
    return product


def get_inverse_matrix(matrix):
    
    determinant = get_determinant(matrix)
    
    # Matrix is not invertible if the determinant is 0
    if determinant == 0:
        return None
    
    cofactor_matrix = []
    n_vec = range(len(matrix))
    for i in n_vec:
        inverse_row = [get_cofactor(matrix, i + 1, j + 1) for j in n_vec]
        cofactor_matrix.append(inverse_row)
    
    adjugate_matrix = get_transpose_matrix(cofactor_matrix)
    
    inverse_matrix = scalar_x_matrix(1 / determinant, adjugate_matrix)
    
    return inverse_matrix



def get_dot_product(vector1, vector2):
    
    if len(vector1) != len(vector2):
        return None
    
    dot_product = 0
    n_vec = range(len(vector1))
    for i in n_vec:
        dot_product += vector1[i] * vector2[i]
    
    return dot_product



def multiply_matrices(matrix1, matrix2):
    
    mat1_n_cols = len(matrix1[0])
    mat2_n_rows = len(matrix2)
    
    # If matrix dimensions don't match, matrixes can't be multiplied
    if mat1_n_cols != mat2_n_rows:
        return None
    
    product_n_rows = len(matrix1)
    product_n_cols = len(matrix2[0])
    
    matrix_product = []
    for i in range(product_n_rows):
        
        row = []
        for j in range(product_n_cols):
            vec1 = matrix1[i]
            vec2 = [matrix2[k][j] for k in range(mat2_n_rows)]
            element = get_dot_product(vec1, vec2)
            row.append(element)
        
        matrix_product.append(row)
        
    return matrix_product
  
  
  
print(get_inverse_matrix(mat3))


# not invertible if determinant = 0
