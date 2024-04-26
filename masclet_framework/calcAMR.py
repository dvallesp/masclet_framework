"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

calcAMR module
Provides functions to perform operations between vector fields
defined on the AMR hierarchy of MASCLET simulations.
Created by Óscar Monllor
"""
import numpy as np
#from numba import njit


############################
# Field (x) Field operations
############################
def add(field1, field2, kept_patches=None):
    '''
    Adds field1 to field2, element-wise.

    Args:
        field1: a list of numpy arrays, each one containing the vector field
                defined on the corresponding grid of the AMR hierarchy

        field2: idem for the second field

        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.

    Returns:

        field: the resulting field with the same structure as the input fields


    Author: Óscar Monllor
    '''

    total_npatch = len(field1)
    assert total_npatch == len(field2), "Field dimensions do not match"

    if kept_patches is None:
        kept_patches = np.ones((total_npatch,), dtype=bool)

    field = []
    for ipatch in range(total_npatch):
        if kept_patches[ipatch]:
            field.append(field1[ipatch] + field2[ipatch])
        else:
            field.append(0)

    return field


def multiply(field1, field2, kept_patches=None):
    '''
    Multiplies field1 by field2, element-wise.

    Args:
        field1: a list of numpy arrays, each one containing the vector field
                defined on the corresponding grid of the AMR hierarchy

        field2: idem for the second field

        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.

    Returns:
        field: the resulting field with the same structure as the input fields

    
    Author: Óscar Monllor
    '''

    total_npatch = len(field1)
    assert total_npatch == len(field2), "Field dimensions do not match"

    if kept_patches is None:
        kept_patches = np.ones((total_npatch,), dtype=bool)

    field = []
    for ipatch in range(total_npatch):
        if kept_patches[ipatch]:
            field.append(field1[ipatch] * field2[ipatch])
        else:
            field.append(0)

    return field


def divide(field1, field2, kept_patches=None):
    '''
    Divides field1 by field2, element-wise.

    Args:
        field1: a list of numpy arrays, each one containing the vector field
                defined on the corresponding grid of the AMR hierarchy

        field2: idem for the second field

        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.

    Returns:
        field: the resulting field with the same structure as the input fields

    
    Author: Óscar Monllor
    '''

    total_npatch = len(field1)
    assert total_npatch == len(field2), "Field dimensions do not match"

    if kept_patches is None:
        kept_patches = np.ones((total_npatch,), dtype=bool)

    field = []
    for ipatch in range(total_npatch):
        if kept_patches[ipatch]:
            field.append(field1[ipatch] / field2[ipatch])
        else:
            field.append(0)

    return field



#############################
# Field (x) Scalar operations
#############################
def add_scalar(field1, scalar, kept_patches=None):
    '''
    Adds a scalar to a field

    Args:
        field1: a list of numpy arrays, each one containing the vector field
                defined on the corresponding grid of the AMR hierarchy

        field2: idem for the second field

        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.

    Returns:
        field: the resulting field with the same structure as the input fields

    
    Author: Óscar Monllor
    '''

    total_npatch = len(field1)
    if kept_patches is None:
        kept_patches = np.ones((total_npatch,), dtype=bool)

    field = []
    for ipatch in range(total_npatch):
        if kept_patches[ipatch]:
            field.append(field1[ipatch] + scalar)
        else:
            field.append(0)

    return field



def multiply_scalar(field1, scalar, kept_patches=None):
    '''
    Multiplies field by a scalar.

    Args:
        field1: a list of numpy arrays, each one containing the vector field
                defined on the corresponding grid of the AMR hierarchy

        field2: idem for the second field

        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.

    Returns:
        field: the resulting field with the same structure as the input fields

    
    Author: Óscar Monllor
    '''

    total_npatch = len(field1)
    if kept_patches is None:
        kept_patches = np.ones((total_npatch,), dtype=bool)
            
    field = []
    for ipatch in range(total_npatch):
        if kept_patches[ipatch]:
            field.append(field1[ipatch] * scalar)
        else:
            field.append(0)
    
    return field

#############################
# Vector Field operations
#############################

def magnitude(field_x, field_y, field_z, kept_patches=None):
    '''
    Calculates the magnitude of a vector field.

    Args:
        field_x: a list of numpy arrays, each one containing the x component of a vector field
                defined on the corresponding grid of the AMR hierarchy

        field_y: idem for the y component of the vector field.
        
        field_z: idem for the z component of the vector field.

        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.

    Returns:
        field: the resulting magnitude of the given vector field with the same structure as the input field

    
    Author: Marco José Molina Pradillo
    '''

    total_npatch = len(field_x)
    assert total_npatch == len(field_y) == len(field_z), "Field dimensions do not match"

    if kept_patches is None:
        kept_patches = np.ones((total_npatch,), dtype=bool)

    field = []
    for ipatch in range(total_npatch):
        if kept_patches[ipatch]:
            mag = np.sqrt(field_x[ipatch]**2 + field_y[ipatch]**2 + field_z[ipatch]**2)
            field.append(mag)
        else:
            field.append(0)

    return field



def magnitude2(field_x, field_y, field_z, kept_patches=None):
    '''
    Calculates the magnitude squared of a vector field.

    Args:
        field_x: a list of numpy arrays, each one containing the x component of a vector field
                defined on the corresponding grid of the AMR hierarchy

        field_y: idem for the y component of the vector field.
        
        field_z: idem for the z component of the vector field.

        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.

    Returns:
        field: the resulting magnitude squared of the given vector field with the same structure as the input field

    
    Author: Marco José Molina Pradillo
    '''

    total_npatch = len(field_x)
    assert total_npatch == len(field_y) == len(field_z), "Field dimensions do not match"

    if kept_patches is None:
        kept_patches = np.ones((total_npatch,), dtype=bool)

    field = []
    for ipatch in range(total_npatch):
        if kept_patches[ipatch]:
            mag_squared = field_x[ipatch]**2 + field_y[ipatch]**2 + field_z[ipatch]**2
            field.append(mag_squared)
        else:
            field.append(0)

    return field