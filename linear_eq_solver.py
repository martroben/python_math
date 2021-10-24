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
  
  
def get_solution_vector(coef_matrix, const_vector):
    return get_matrix_product(get_inverse_matrix(coef_matrix), const_vector)

  
  
###################################

######################
# General operations #
######################

# Get i,j-th element of a square matrix (indexing starting from 0)
def get_matrix_element(matrix, coordinates):
    return matrix[coordinates[0]][coordinates[1]]
 

# Product of all elements in list
def multiply_list(list):
    
    product = 1
    for i in list:
        product *= i
        
    return product


# Multiply all elements of a vector (list) by a number
def scalar_x_vector(scalar, vector):
    return [scalar * vector[i] for i in range(len(vector))]

  
  
########################
# Finding combinations #
########################

# the algorithm of finding next value in combinations:
# Use the next biggest value in sequence to replace first (boundary) value
# Then arrange others in ascending order
# For example:
# [1, 3, 2, 0] --> [2, 0, 1, 3]
# [3, 4, 1] --> [4, 1, 3]
# [0, 1] --> [1, 0]
def next_up(input):
    
    old_boundary_rank = sorted(input).index(input[0])
    new_boundary_value = sorted(input)[old_boundary_rank + 1]
    other_values = sorted(input)
    other_values.remove(new_boundary_value)
    
    return [new_boundary_value] + other_values
    

# Get combinations using "the algorithm"
# Start from the biggest value and move up/down the combination to find the boundary of last descending values (or last value)
# Rearrange descending part + boundary value according to the next_up algorithm
# For example:
# [4, 3, 0, 2, 1] - the boundary is on 0 (because 2 and 1 are in descending order) --> [4, 3, 1, 0, 2]
# [0, 1, 2, 3, 4] - the boundary is on 3 --> [0, 1, 2, 4, 3]
# [4, 2, 0, 3, 1] - boundary on 0 --> [4, 2, 1, 0, 3]
# [4, 1, 3, 2, 0] - boundary on 1 --> [4, 2, 0, 1, 3]
def get_next_combination(input):
    
    marker = input.index(max(input))
    
    if input[marker:] == sorted(input[marker:], reverse = True):
        
        while input[marker:] == sorted(input[marker:], reverse = True):
            marker -= 1
    else:
        while input[(marker + 1):] != sorted(input[(marker + 1):], reverse = True):
            marker += 1
    
    return input[:marker] + next_up(input[marker:])


# Non-recursive function to get combinations
# Recursive function can only handle combinations of up to 6 elements, because 6! = 720
# Pyhon default maximum recursive depth is 1000
def get_combinations(n):

    start_vec = [i for i in range(n)]
    end_vec = sorted(start_vec, reverse = True)
    combinations = [start_vec]
    
    while combinations[-1] != end_vec:
        combinations.append(get_next_combination(combinations[-1]))
    
    return combinations

  

###########################################
# Calculating determinant non-recursively #
###########################################

# Turn combination into coordinates that make up addends in determinant calculation
# Eg. [2, 3, 0, 1] --> [02, 13, 20, 31] --> [[0, 2], [1, 3], [2, 0], [3, 1]]
def make_determinant_coordinates(combination):
    
    coordinates = []
    for i in range(len(combination)):
        coordinates.append([i, combination[i]])
    
    return coordinates


# Calculate the order of signs of factors in determinant sum
# Only works if addend order in in the order in which combinations are generated by get_combinations()
def get_determinant_signs(n):

    signs_vec = [1]
    for s in range(1, n):
        seed = [(-1) ** i for i in range(s+1)]
        
        new_iteration = []
        for t in seed:
            new_iteration += scalar_x_vector(t, signs_vec)
    
        signs_vec = new_iteration
    
    return signs_vec
  

# 1. Gets combinations of matrix elements that make up factors in determinant sum
# 2. Gets signs of these factors
# 3. Adds all up
# Eg.
#          | a11  a12  a13 |
# Matrix:  | a21  a22  a23 |
#          | a31  a32  a33 |
#
# Factors: a11 a22 a33 | a11 a23 a32 | a12 a21 a33 | a12 a23 a31 | a13 a21 a32 | a13 a22 a31
# Determinant sum (with signs): a11 a22 a33 - a11 a23 a32 - a12 a21 a33 + a12 a23 a31 + a13 a21 a32 - a13 a22 a31
def get_determinant_nonrecursive(matrix):
    
    if len(matrix) == 1:
        return matrix[0][0]

    combinations = get_combinations(len(matrix))
    det_coordinates = [make_determinant_coordinates(cmb) for cmb in combinations]

    det_signs = get_determinant_signs(len(matrix))
    
    determinant = 0
    for i in range(len(det_coordinates)):
        factors = [get_matrix_element(matrix, element) for element in det_coordinates[i]]
        determinant += det_signs[i] * multiply_list(factors)

    return determinant 

  
  

def factorial(n):

    result = 1
    for i in range(2, n + 1):
        result *= i
        
    return result

  
  
print(get_inverse_matrix(mat3))


# not invertible if determinant = 0
