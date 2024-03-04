"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

calcAMR module
Provides functions to perform operations between vector fields
defined on the AMR hierarchy of MASCLET simulations.
Created by Óscar Monllor
"""
############################
# Field (x) Field operations
############################
def add(field1, field2):
    '''
    Adds field1 to field2, element-wise.

    Args:
        field1: a list of numpy arrays, each one containing the vector field
                defined on the corresponding grid of the AMR hierarchy

        field2: idem for the second field

    Returns:

        field: the resulting field with the same structure as the input fields


    Author: Óscar Monllor
    '''

    return [field1[ipatch] + field2[ipatch] for ipatch in range(len(field1))]

def multiply(field1, field2):
    '''
    Multiplies field1 by field2, element-wise.

    Args:
        field1: a list of numpy arrays, each one containing the scalar field
                defined on the corresponding grid of the AMR hierarchy

        field2: idem for the second field

    Returns:
        field: the resulting field with the same structure as the input fields


     Author: Óscar Monllor
    '''

    return [field1[ipatch]*field2[ipatch] for ipatch in range(len(field1))]


def divide(field1, field2):
    '''
    Divides field1 by field2, element-wise.

    Args:
        field1: a list of numpy arrays, each one containing the scalar field
                defined on the corresponding grid of the AMR hierarchy

        field2: idem 

    Returns:
        field: the resulting field with the same structure as the input fields


    Author: Óscar Monllor
    '''

    return [field1[ipatch]/field2[ipatch] for ipatch in range(len(field1))]

#############################
# Field (x) Scalar operations
#############################
def add_scalar(field1, scalar):
    '''
    Adds a scalar to a field

    Args:
        field1: a list of numpy arrays, each one containing the scalar field
                defined on the corresponding grid of the AMR hierarchy

        scalar: the scalar to add to the field

    Returns:

        field: the resulting field with the same structure as the input field


    Author: Óscar Monllor
    '''

    return [field1[ipatch] + scalar for ipatch in range(len(field1))]



def multiply_scalar(field1, scalar):
    '''
    Multiplies field by a scalar.

    Args:
        field1: a list of numpy arrays, each one containing the scalar field
                defined on the corresponding grid of the AMR hierarchy

        scalar: the scalar to multiply the field by

    Returns:
        field: the resulting field with the same structure as the input field

    
    Author: Óscar Monllor
    '''
            
    return [field1[ipatch]*scalar for ipatch in range(len(field1))]