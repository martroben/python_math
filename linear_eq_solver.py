#######################################################################################################
##                                                                                                   ##
##  Script name: linear_eq_solver.py                                                                 ##
##  Purpose of script: Linear equation solver for a Python CodeClub Estonia challenge                ##
##  Author: Mart Roben                                                                               ##
##  Date Created: 3 Nov 2021                                                                         ##
##                                                                                                   ##
##  Copyright: BSD-3-Clause                                                                          ##
##  https://github.com/martroben/linear_eq_solver                                                    ##
##                                                                                                   ##
##  Contact: fb.com/mart.roben                                                                       ##
##                                                                                                   ##
##  Challenge description:                                                                           ##
##  https://pyspace.eu/ws/thorgate/ch/23/                                                            ##
##                                                                                                   ##
##  Write a function solve(*equations) that can solve systems of linear equations                    ##
##  with 2, 3 to 6 variables.                                                                        ##
##  Equations will look like this: '2x+3y=6', '4x+9y=15'                                             ##
##  Return a dictionary with the results, i.e. {'x': 1.5, 'y': 1}.                                   ##
##  When solving higher order equation systems, variables can be named by any letter from a to z.    ##
##  You can assume that all equations are refined as much as possible so you don't have to handle    ##
##  cases like 2x+5x-2x-55y=2 but it will be in the form 5x-55y=2.                                   ##
##  However, quantifiers before variables might include floats so you have to handle                 ##
##  cases like 0.5x+2y-4.6z=-1555.5.                                                                 ##
##  When dealing with floats, round the number to 2 decimal places so that the tester would accept   ##
##  your result. I.e. 2.55935312 -> 2.56.                                                            ##
##  NB! numpy is not allowed.                                                                        ##
##                                                                                                   ##
##  Extra challenges for fun:                                                                        ##
##  No imports                                                                                       ##
##  Handle linear equations with more than 6 variables (ie. no recursion)                            ##
##  Handle input ["2x+3l=6", "-z + j = - 7", "3x + 0.002k = 0", "1/2y-x/2=3/4", "5*x = .1",          ##
##                "m/-3 + z*2 + .4*z = 3/-4", "j = 3j - k"]                                          ##
##                                                                                                   ##
#######################################################################################################


###########################
# Input parsing functions #
###########################

# Return characters from list of strings that are not alphanumeric and not in known_chars
def unknown_input_chars(str_list, known_chars):
    
    unknown_chars = []
    for str in str_list:
        for char in str:
            if not(char in known_chars or char.isalnum()):
                unknown_chars += char
    
    return unknown_chars


# Parse a multiplier (number) from a given position of input string
def get_multiplier_at(i_pos, str, decimal_chars):
    
    multiplier = ""
    
    while i_pos < len(str) \
        and (str[i_pos].isdigit() \
        or (str[i_pos] in decimal_chars)):
        multiplier += str[i_pos]
        i_pos += 1
        
    return multiplier


# Parse a variable name (character string) from a given position of input string
def get_variable_at(i_pos, str):

    variable = ""
    
    while i_pos < len(str) and str[i_pos].isalnum():
        
        variable += str[i_pos]
        i_pos += 1
        
    return variable


# Parse sign, multiplier and variable name for each member of a single side of equation
# Eg. "2x - 3y" --> [[1, 2, "x"], [-1, 3, "y"]]
def parse_members(expression):
    
    # [0]: sign, [1]: multiplier, [2]: variable name
    empty_member = [1, 1, ""]
    
    members = [empty_member.copy()]
    i_list = 0
    i_pos = 0
    
    while i_pos in range(len(expression)):   
        if expression[i_pos] in multiplication_chars:
            
            reciprocal = expression[i_pos] == "/"
            
            if expression[i_pos+1] in negative_chars:
                members[i_list][0] *= -1
                i_pos += 1
            
            i_pos += 1
            
            if expression[i_pos].isdigit() or (expression[i_pos] in decimal_chars):
                multiplier = get_multiplier_at(i_pos, expression, decimal_chars)
                multiplier_float = float(multiplier.replace(",", "."))
                if reciprocal: multiplier_float = 1 / multiplier_float
                members[i_list][1] *= multiplier_float
                i_pos += len(multiplier)
                
            elif expression[i_pos].isalpha():
                variable = get_variable_at(i_pos, expression)
                members[i_list][2] = variable
                i_pos += len(variable)
                
            else:
                print("Error parsing expression '" + expression + "'!")
                break
            
        elif expression[i_pos].isdigit() or (expression[i_pos] in decimal_chars):
            multiplier = get_multiplier_at(i_pos, expression, decimal_chars)
            multiplier_float = float(multiplier.replace(",", "."))
            members[i_list][1] *= multiplier_float
            i_pos += len(multiplier)
            
        elif expression[i_pos].isalpha():
            variable = get_variable_at(i_pos, expression)
            members[i_list][2] = variable
            i_pos += len(variable)
            
        elif expression[i_pos] in addition_chars:

            if i_pos != 0:
                members += [empty_member.copy()]
                i_list += 1
            
            if expression[i_pos] in negative_chars:
                members[i_list][0] *= -1
                
            i_pos += 1
        
        else:
            print("Error parsing expression '" + expression + "'!")
            break

    return members


# Consolidate left and right side of equation to a {variable: multiplier} dict
# Eg. [[1, 2, "x"], [-1, 3, "y"]], [[1, 5, ]] --> {"x": 2, "y": -3, "": -5}
def consolidate_members(member_list, right_side = False):
    
    member_dict = {}
    
    for member in member_list:
        if member[2] not in member_dict:
            member_dict[member[2]] = member[0] * member[1] * (-1)**right_side
        else:
            member_dict[member[2]] += member[0] * member[1] * (-1)**right_side
        
    return member_dict


# Parse equation string to a {variable: multiplier} dict
# Eg. "2x - 3y = 5" --> {"x": 2, "y": -3, "": -5}
def parse_equation(equation_str):
    
    # Remove whitespaces
    for character in space_chars:
        equation_str = equation_str.replace(character, "")
    
    equation_sides = equation_str.split("=")
    
    eq_left_side = consolidate_members(parse_members(equation_sides[0]))
    eq_right_side = consolidate_members(parse_members(equation_sides[1]), right_side = True)
    
    unique_variables = set(list(eq_left_side.keys()) + list(eq_right_side.keys()))
    
    eq_standardized = {}
    
    for key in unique_variables:
        eq_standardized[key] = sum([x for x in [eq_left_side.get(key), eq_right_side.get(key)] if x != None])
        
    return eq_standardized


# Find all unique variables and add zeros for missing variables in each equation
# Eg. [{"a": 2, "": 5}, {"b": 3, "": 10}] --> [{"a": 2, "": 5, "b": 0}, {"b": 3, "": 10, "a": 0}]
def standardize_equations(eq_sys):

    unique_variables = set()
    for eq in eq_sys:
        for key in eq.keys():
            unique_variables.add(key)

    for eq in eq_sys:
        for var in unique_variables:
            if var not in eq:
                eq[var] = 0

    return eq_sys




######################
# General operations #
######################

# Product of all elements in list
def multiply_list(list):
    
    product = 1
    for i in list:
        product *= i
        
    return product


# Multiply all elements of a vector (list) by a number
def scalar_x_vector(scalar, vector):
    return [scalar * vector[i] for i in range(len(vector))]


# Multiply all elements of a square matrix by a number
def scalar_x_matrix(scalar, matrix):
    
    product = []
    n_vec = range(len(matrix))
    for i in n_vec:
        product_row = [scalar * matrix[i][j] for j in n_vec]
        product.append(product_row)
    
    return product


# Dot product of two vectors
def get_dot_product(vector1, vector2):
    
    if len(vector1) != len(vector2):
        return None
    
    dot_product = 0
    n_vec = range(len(vector1))
    for i in n_vec:
        dot_product += vector1[i] * vector2[i]
    
    return dot_product


# Factorial of a number
def factorial(n):

    result = 1
    for i in range(2, n + 1):
        result *= i
        
    return result

  
 
  
#####################
# Matrix operations #
#####################

# Get i,j-th element of a square matrix (indexing starting from 0)
# coordinates variable is a list [i, j]
def get_matrix_element(matrix, coordinates):
    return matrix[coordinates[0]][coordinates[1]]

  
def get_transpose_matrix(matrix):
    
    matrix_T = []
    n_vec = range(len(matrix))
    for i in n_vec:
        row_T = [matrix[j][i] for j in n_vec]
        matrix_T.append(row_T)
    
    return matrix_T


# Get i,j submatrix
# i - rows, j - cols (indexing starting from 1)
def get_submatrix(matrix, i, j):
    
    submatrix = []
    for k in range(len(matrix)):
        submatrix.append(matrix[k][:(j-1)] + matrix[k][j:])
    
    submatrix.pop(i-1)
    
    return submatrix


# Get i,j cofactor
# indexing starting from 1
def get_cofactor(matrix, i, j):
    
    submatrix = get_submatrix(matrix, i, j)
    cofactor_sign = (-1) ** (i + j)
    cofactor = cofactor_sign * get_determinant(submatrix)
    
    return cofactor


# Recursive function to calculate determinants
# Can handle up to 6x6 matrices
# Replaced by non-recursive version
# def get_determinant_recursive(matrix):
#     
#     if len(matrix) == 1:
#         return matrix[0][0]
# 
#     determinant = 0
#     for n in range(len(matrix)):
#         determinant += matrix[0][n] * get_cofactor(matrix, i = 1, j = n + 1)
# 
#     return determinant


# Invert matrix:
# Multiply transpose of cofactor matrix with reciprocal of the determinant
def get_inverse_matrix(matrix, determinant = None):
    
    if determinant == None:
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


# Multiply two matrices (don't have to be square)
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


 
  
########################
# Finding combinations #
########################

# the algorithm of finding next value in combinations:
# Use the next biggest value in sequence to replace first (boundary) value
# Then arrange others in ascending order
# Eg.
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
# Eg.
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
#
# Eg.
#          | a11  a12  a13 |
# Matrix:  | a21  a22  a23 |
#          | a31  a32  a33 |
#
# Determinant factors: a11 a22 a33 | a11 a23 a32 | a12 a21 a33 | a12 a23 a31 | a13 a21 a32 | a13 a22 a31
# Determinant sum (with signs): a11*a22*a33 - a11*a23*a32 - a12*a21*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31

def get_determinant(matrix):
    
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




###########################
# Solving linear equation #
###########################

# Checks if matrix is a square (nxn) matrix
def is_square(coef_matrix):
    
    square = True
    for row in coef_matrix:
        if len(row) != len(coef_matrix):
            square = False
        
    return square


# Get solutions of linear equation by multiplying inverse to coefficients matrix with constants vector
def get_solution_vector(coef_matrix, const_vector, determinant = None):
    return multiply_matrices(get_inverse_matrix(coef_matrix, determinant), const_vector)


# Check if calculated values give the correct answer in linear equations
def check_solution(coefs, consts, solution):
    return [ round(i[0], 10) for i in multiply_matrices(coefs, solution) ] == [ round(i[0], 10) for i in consts ]


# Turn solution vector to {var_name: solution} dictionary
def solution_out(sol_vec, var_indexes):
    
    out = {}
    for i, value in enumerate(sol_vec):
        out[var_indexes[i]] = round(value[0], 2)
    
    return out


# Take equation strings as arguments and return solution as a {var_name: solution} dictionary
def solve(*equations):
    
    # Handle the case when input is a list of equation strings
    if len(equations) == 1 and type(equations[0]) is list:
        equations = equations[0]

    input = [ i for i in equations]

    bad_chars = unknown_input_chars(input, known_characters)
    if len(bad_chars) > 0:
        print("Unrecognized characters found in input: " + str(bad_chars))
        
        return
    
    # Get list of {variable: multiplier} dictionaries for each input equation
    system_of_equations = list(map(parse_equation, input))
    
    
    # Standardize equations
    system_of_equations = standardize_equations(system_of_equations)
    
    
    # Separate constants to a vector and remove them from equations
    constants_vector = [ [-1 * dict[""]] for dict in system_of_equations ]
    
    only_variables = [eq.copy() for eq in system_of_equations]
    for expression in only_variables:
        del expression[""]
    
    
    # Create a key-index dictionary for positions of variable names in matrix
    variable_indexes = {}
    for i, var in enumerate(sorted(list(only_variables[0].keys()))):
        variable_indexes[i] = var
    
    
    # Collect coefficients to a matrix
    coefficients_matrix = []
    for expression in only_variables:
        coefficients_matrix += [[ expression[variable_indexes[i]] for i in range(len(variable_indexes)) ]]
    
    
    # Check solvability, solve linear equation, check soluton and format output to a {var_name: solution} dictionary
    if is_square(coefficients_matrix):
        determinant = get_determinant(coefficients_matrix)
        
        if determinant != 0:
            solution_vec = get_solution_vector(coefficients_matrix, constants_vector, determinant)
            
            if check_solution(coefficients_matrix, constants_vector, solution_vec):
                output = solution_out(solution_vec, variable_indexes)
                
                return output
            
            else:
                print("Wrong solution! Found a solution but it doesn't seem to solve the equations:")
                print(solution_out(solution_vec, variable_indexes))
            
        else:
            print("No solution found for this system of equations! Determinant = 0:")
            print(input)
            
    else:
        print("No solution found for this system of equations! Doesn't give a square (nxn) coefficients matrix:")
        print(input)




####################
# Global variables #
####################

decimal_chars = [".", ","]
addition_chars = ["+", "-", "–"]
negative_chars = ["-", "–"]
multiplication_chars = ["*", "/"]
space_chars = [" ", "_"]
known_characters = decimal_chars + addition_chars + negative_chars + multiplication_chars + space_chars + ["="]




##########
# ON RUN #
##########

# Test inputs
test_input1 = ["2x+3l=6", "-z + j = - 7", "3x + 0.002k = 0", "1/2y-x/2=3/4", "5*x = .1", "m/-3 + z*2 + .4*z = 3/-4", "j = 3j - k"]

test_input2 = ["2x + y = 3",
               "-.5x = 1"]

test_input3 = ["3x + y = 0", "4z = 8"]

test_input4 = ["3a + 5b - 23c + 9e + f + 8.5g + 0.5h + 5i = 40",
               "32a + b + 10c + 2e - 15f + 6g + 22h + 9i = -2",
               "-a + 16b + 5c - 19e + 2f - 8g + 11h - 2i = 52",
               "30a - 2b + 9c + 4e - 32f + 11g + 35h + 26i = 3",
               "24a + 12b + 7c + 38e - 2f + 34g - 28h + 7i = 21",
               "a + 22b - 3c + 19e + 8f - 4g + 41h + 13i = 0",
               "8a + 31b + 4c + 41e + 3f + 5g + 25h + i = 11",
               "-32a + 2b - c + 5e + 24f - 3g + 22h + 17i = 9"]

test_input5 = "3a + 3a = -3"

test_input6 = ["&/(.0af!", "34x + 2 ' 3"]

test_input7 = ["x + 2y + 3z = 1", "2x + 4y + 6z = 4", "-3x - 6y + z = 11"]

# Call function
solve(test_input1)

