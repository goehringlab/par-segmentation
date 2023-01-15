"""
tfg.math.interpolation.bspline.interpolate

"""

from typing import Union, Sequence, Tuple
import numpy as np
import tensorflow as tf
from absl import flags
import enum

FLAGS = flags.FLAGS

Integer = Union[int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                np.uint32, np.uint64]
Float = Union[float, np.float16, np.float32, np.float64]
TensorLike = Union[Integer, Float, Sequence, np.ndarray, tf.Tensor, tf.Variable]


def interpolate(knots: TensorLike,
                positions: TensorLike,
                degree: int,
                cyclical: bool,
                name: str = "bspline_interpolate") -> tf.Tensor:
    """Applies B-spline interpolation to input control points (knots).
    Note:
      In the following, A1 to An, and B1 to Bk are optional batch dimensions.
    Args:
      knots: A tensor with shape `[B1, ..., Bk, C]` containing knot values, where
        `C` is the number of knots.
      positions: Tensor with shape `[A1, .. An]`. Positions must be between
        `[0, C - D)` for non-cyclical and `[0, C)` for cyclical splines, where `C`
        is the number of knots and `D` is the spline degree.
      degree: An `int` between 0 and 4, or an enumerated constant from the Degree
        class, which is the degree of the splines.
      cyclical: A `bool`, whether the splines are cyclical.
      name: A name for this op. Defaults to "bspline_interpolate".
    Returns:
      A tensor of shape `[A1, ... An, B1, ..., Bk]`, which is the result of spline
      interpolation.
    """
    with tf.name_scope(name):
        knots = tf.convert_to_tensor(value=knots)
        positions = tf.convert_to_tensor(value=positions)

        num_knots = knots.get_shape().as_list()[-1]
        weights = knot_weights(positions, num_knots, degree, cyclical, False, name)
        return interpolate_with_weights(knots, weights)


def interpolate_with_weights(
        knots: TensorLike,
        weights: TensorLike,
        name: str = "bspline_interpolate_with_weights") -> tf.Tensor:
    """Interpolates knots using knot weights.
    Note:
      In the following, A1 to An, and B1 to Bk are optional batch dimensions.
    Args:
      knots: A tensor with shape `[B1, ..., Bk, C]` containing knot values, where
        `C` is the number of knots.
      weights: A tensor with shape `[A1, ..., An, C]` containing dense weights for
        the knots, where `C` is the number of knots.
      name: A name for this op. Defaults to "bspline_interpolate_with_weights".
    Returns:
      A tensor with shape `[A1, ..., An, B1, ..., Bk]`, which is the result of
      spline interpolation.
    Raises:
      ValueError: If the last dimension of knots and weights is not equal.
    """
    with tf.name_scope(name):
        knots = tf.convert_to_tensor(value=knots)
        weights = tf.convert_to_tensor(value=weights)

        compare_dimensions(
            tensors=(knots, weights), axes=-1, tensor_names=("knots", "weights"))

    return tf.tensordot(weights, knots, (-1, -1))


def compare_dimensions(tensors, axes, tensor_names=None):
    """Compares dimensions of tensors with static or dynamic shapes.
    Args:
      tensors: A list or tuple of tensors to compare.
      axes: An `int` or a list or tuple of `int`s with the same length as
        `tensors`. If an `int`, it is assumed to be the same for all the tensors.
        Each entry should correspond to the axis of the tensor being compared.
      tensor_names: Names of `tensors` to be used in the error message if one is
        thrown. If left as `None`, their `Tensor.name` fields are used instead.
    Raises:
      ValueError: If inputs have unexpected types, or if given axes are out of
        bounds, or if the check fails.
    """
    _check_tensors(tensors, 'tensors')
    if isinstance(axes, int):
        axes = [axes] * len(tensors)
    _check_tensor_axis_lists(tensors, 'tensors', axes, 'axes')
    axes = _fix_axes(tensors, axes, allow_negative=False)
    if tensor_names is None:
        tensor_names = _give_default_names(tensors, 'tensor')
    dimensions = [_get_dim(tensor, axis) for tensor, axis in zip(tensors, axes)]
    if not _all_are_equal(dimensions):
        raise ValueError('Tensors {} must have the same number of dimensions in '
                         'axes {}, but they are {}.'.format(
            list(tensor_names), list(axes), list(dimensions)))


def _all_are_equal(list_of_objects):
    """Helper function to check if all the items in a list are the same."""
    if not list_of_objects:
        return True
    if isinstance(list_of_objects[0], list):
        list_of_objects = [tuple(obj) for obj in list_of_objects]
    return len(set(list_of_objects)) == 1


def _get_dim(tensor, axis):
    """Returns dimensionality of a tensor for a given axis."""
    return tf.compat.dimension_value(tensor.shape[axis])


def _check_tensors(tensors, tensors_name):
    """Helper function to check the type and length of tensors."""
    _check_type(tensors, tensors_name, (list, tuple))
    if len(tensors) < 2:
        raise ValueError('At least 2 tensors are required.')


def _check_type(variable, variable_name, expected_type):
    """Helper function for checking that inputs are of expected types."""
    if isinstance(expected_type, (list, tuple)):
        expected_type_name = 'list or tuple'
    else:
        expected_type_name = expected_type.__name__
    if not isinstance(variable, expected_type):
        raise ValueError('{} must be of type {}, but it is {}'.format(
            variable_name, expected_type_name,
            type(variable).__name__))


def _check_tensor_axis_lists(tensors, tensors_name, axes, axes_name):
    """Helper function to check that lengths of `tensors` and `axes` match."""
    _check_type(axes, axes_name, (list, tuple))
    if len(tensors) != len(axes):
        raise ValueError(
            '{} and {} must have the same length, but are {} and {}.'.format(
                tensors_name, axes_name, len(tensors), len(axes)))


def _fix_axes(tensors, axes, allow_negative):
    """Makes all axes positive and checks for out of bound errors."""
    axes = [
        axis + tensor.shape.ndims if axis < 0 else axis
        for tensor, axis in zip(tensors, axes)
    ]
    if not all(
            ((allow_negative or
              (not allow_negative and axis >= 0)) and axis < tensor.shape.ndims)
            for tensor, axis in zip(tensors, axes)):
        rank_axis_pairs = list(
            zip([tensor.shape.ndims for tensor in tensors], axes))
        raise ValueError(
            'Some axes are out of bounds. Given rank-axes pairs: {}'.format(
                [pair for pair in rank_axis_pairs]))
    return axes


def _give_default_names(list_of_objects, name):
    """Helper function to give default names to objects for error messages."""
    return [name + '_' + str(index) for index in range(len(list_of_objects))]


class Degree(enum.IntEnum):
    """Defines valid degrees for B-spline interpolation."""
    CONSTANT = 0
    LINEAR = 1
    QUADRATIC = 2
    CUBIC = 3
    QUARTIC = 4


def _constant(position: tf.Tensor) -> tf.Tensor:
    """B-Spline basis function of degree 0 for positions in the range [0, 1]."""
    # A piecewise constant spline is discontinuous at the knots.
    return tf.expand_dims(tf.clip_by_value(1.0 + position, 1.0, 1.0), axis=-1)


def _linear(position: tf.Tensor) -> tf.Tensor:
    """B-Spline basis functions of degree 1 for positions in the range [0, 1]."""
    # Piecewise linear splines are C0 smooth.
    return tf.stack((1.0 - position, position), axis=-1)


def _quadratic(position: tf.Tensor) -> tf.Tensor:
    """B-Spline basis functions of degree 2 for positions in the range [0, 1]."""
    # We pre-calculate the terms that are used multiple times.
    pos_sq = tf.pow(position, 2.0)

    # Piecewise quadratic splines are C1 smooth.
    return tf.stack((tf.pow(1.0 - position, 2.0) / 2.0, -pos_sq + position + 0.5,
                     pos_sq / 2.0),
                    axis=-1)


def _cubic(position: tf.Tensor) -> tf.Tensor:
    """B-Spline basis functions of degree 3 for positions in the range [0, 1]."""
    # We pre-calculate the terms that are used multiple times.
    neg_pos = 1.0 - position
    pos_sq = tf.pow(position, 2.0)
    pos_cb = tf.pow(position, 3.0)

    # Piecewise cubic splines are C2 smooth.
    return tf.stack(
        (tf.pow(neg_pos, 3.0) / 6.0, (3.0 * pos_cb - 6.0 * pos_sq + 4.0) / 6.0,
         (-3.0 * pos_cb + 3.0 * pos_sq + 3.0 * position + 1.0) / 6.0,
         pos_cb / 6.0),
        axis=-1)


def _quartic(position: tf.Tensor) -> tf.Tensor:
    """B-Spline basis functions of degree 4 for positions in the range [0, 1]."""
    # We pre-calculate the terms that are used multiple times.
    neg_pos = 1.0 - position
    pos_sq = tf.pow(position, 2.0)
    pos_cb = tf.pow(position, 3.0)
    pos_qt = tf.pow(position, 4.0)

    # Piecewise quartic splines are C3 smooth.
    return tf.stack(
        (tf.pow(neg_pos, 4.0) / 24.0,
         (-4.0 * tf.pow(neg_pos, 4.0) + 4.0 * tf.pow(neg_pos, 3.0) +
          6.0 * tf.pow(neg_pos, 2.0) + 4.0 * neg_pos + 1.0) / 24.0,
         (pos_qt - 2.0 * pos_cb - pos_sq + 2.0 * position) / 4.0 + 11.0 / 24.0,
         (-4.0 * pos_qt + 4.0 * pos_cb + 6.0 * pos_sq + 4.0 * position + 1.0) /
         24.0, pos_qt / 24.0),
        axis=-1)


def knot_weights(
        positions: TensorLike,
        num_knots: TensorLike,
        degree: int,
        cyclical: bool,
        sparse_mode: bool = False,
        name: str = "bspline_knot_weights") -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
    """Function that converts cardinal B-spline positions to knot weights.
    Note:
      In the following, A1 to An are optional batch dimensions.
    Args:
      positions: A tensor with shape `[A1, .. An]`. Positions must be between
        `[0, C - D)` for non-cyclical and `[0, C)` for cyclical splines, where `C`
        is the number of knots and `D` is the spline degree.
      num_knots: A strictly positive `int` describing the number of knots in the
        spline.
      degree: An `int` describing the degree of the spline, which must be smaller
        than `num_knots`.
      cyclical: A `bool` describing whether the spline is cyclical.
      sparse_mode: A `bool` describing whether to return a result only for the
        knots with nonzero weights. If set to True, the function returns the
        weights of only the `degree` + 1 knots that are non-zero, as well as the
        indices of the knots.
      name: A name for this op. Defaults to "bspline_knot_weights".
    Returns:
      A tensor with dense weights for each control point, with the shape
      `[A1, ... An, C]` if `sparse_mode` is False.
      Otherwise, returns a tensor of shape `[A1, ... An, D + 1]` that contains the
      non-zero weights, and a tensor with the indices of the knots, with the type
      tf.int32.
    Raises:
      ValueError: If degree is greater than 4 or num_knots - 1, or less than 0.
      InvalidArgumentError: If positions are not in the right range.
    """
    with tf.name_scope(name):
        positions = tf.convert_to_tensor(value=positions)

        if degree > 4 or degree < 0:
            raise ValueError("Degree should be between 0 and 4.")
        if degree > num_knots - 1:
            raise ValueError("Degree cannot be >= number of knots.")
        if cyclical:
            positions = assert_all_in_range(positions, 0.0, float(num_knots))
        else:
            positions = assert_all_in_range(positions, 0.0,
                                            float(num_knots - degree))

        all_basis_functions = {
            # Maps valid degrees to functions.
            Degree.CONSTANT: _constant,
            Degree.LINEAR: _linear,
            Degree.QUADRATIC: _quadratic,
            Degree.CUBIC: _cubic,
            Degree.QUARTIC: _quartic
        }
        basis_functions = all_basis_functions[degree]

        if not cyclical and num_knots - degree == 1:
            # In this case all weights are non-zero and we can just return them.
            if not sparse_mode:
                return basis_functions(positions)
            else:
                shift = tf.zeros_like(positions, dtype=tf.int32)
                return basis_functions(positions), shift

        # shape_batch = positions.shape.as_list()
        shape_batch = tf.shape(input=positions)
        positions = tf.reshape(positions, shape=(-1,))

        # Calculate the nonzero weights from the decimal parts of positions.
        shift = tf.floor(positions)
        sparse_weights = basis_functions(positions - shift)
        shift = tf.cast(shift, tf.int32)

        if sparse_mode:
            # Returns just the weights and the shift amounts, so that tf.gather_nd on
            # the knots can be used to sparsely activate knots if needed.
            shape_weights = tf.concat(
                (shape_batch, tf.constant((degree + 1,), dtype=tf.int32)), axis=0)
            sparse_weights = tf.reshape(sparse_weights, shape=shape_weights)
            shift = tf.reshape(shift, shape=shape_batch)
            return sparse_weights, shift

        num_positions = tf.size(input=positions)
        ind_row, ind_col = tf.meshgrid(
            tf.range(num_positions, dtype=tf.int32),
            tf.range(degree + 1, dtype=tf.int32),
            indexing="ij")

        tiled_shifts = tf.reshape(
            tf.tile(tf.expand_dims(shift, axis=-1), multiples=(1, degree + 1)),
            shape=(-1,))
        ind_col = tf.reshape(ind_col, shape=(-1,)) + tiled_shifts
        if cyclical:
            ind_col = tf.math.mod(ind_col, num_knots)
        indices = tf.stack((tf.reshape(ind_row, shape=(-1,)), ind_col), axis=-1)
        shape_indices = tf.concat((tf.reshape(
            num_positions, shape=(1,)), tf.constant(
            (degree + 1, 2), dtype=tf.int32)),
            axis=0)
        indices = tf.reshape(indices, shape=shape_indices)
        shape_scatter = tf.concat((tf.reshape(
            num_positions, shape=(1,)), tf.constant((num_knots,), dtype=tf.int32)),
            axis=0)
        weights = tf.scatter_nd(indices, sparse_weights, shape_scatter)
        shape_weights = tf.concat(
            (shape_batch, tf.constant((num_knots,), dtype=tf.int32)), axis=0)
        return tf.reshape(weights, shape=shape_weights)


TFG_ADD_ASSERTS_TO_GRAPH = 'tfg_add_asserts_to_graph'


def assert_all_in_range(vector,
                        minval,
                        maxval,
                        open_bounds=False,
                        name='assert_all_in_range'):
    """Checks whether all values of vector are between minval and maxval.
    This function checks if all the values in the given vector are in an interval
    `[minval, maxval]` if `open_bounds` is `False`, or in `]minval, maxval[` if it
    is set to `True`.
    Note:
      In the following, A1 to An are optional batch dimensions.
    Args:
      vector: A tensor of shape `[A1, ..., An]` containing the values we want to
        check.
      minval: A `float` or a tensor of shape `[A1, ..., An]` representing the
        desired lower bound for the values in `vector`.
      maxval: A `float` or a tensor of shape `[A1, ..., An]` representing the
        desired upper bound for the values in `vector`.
      open_bounds: A `bool` indicating whether the range is open or closed.
      name: A name for this op. Defaults to 'assert_all_in_range'.
    Raises:
      tf.errors.InvalidArgumentError: If `vector` is not in the expected range.
    Returns:
      The input vector, with dependence on the assertion operator in the graph.
    """

    # if not FLAGS[TFG_ADD_ASSERTS_TO_GRAPH].value:
    #     return vector

    # return vector

    with tf.name_scope(name):
        vector = tf.convert_to_tensor(value=vector)
        minval = tf.convert_to_tensor(value=minval, dtype=vector.dtype)
        maxval = tf.convert_to_tensor(value=maxval, dtype=vector.dtype)

        if open_bounds:
            assert_ops = (tf.debugging.assert_less(vector, maxval),
                          tf.debugging.assert_greater(vector, minval))
        else:
            assert_ops = (tf.debugging.assert_less_equal(vector, maxval),
                          tf.debugging.assert_greater_equal(vector, minval))
        with tf.control_dependencies(assert_ops):
            return tf.identity(vector)
