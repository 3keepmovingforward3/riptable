import builtins # Necessary because of the "from riptable import *" below.
from enum import IntEnum
import operator
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np

import pytest
from numpy.testing import assert_array_equal

import riptable as rt
from riptable import FastArray, Categorical, CatZero
from riptable.rt_categorical import Categories
from riptable.rt_enum import (
    INVALID_DICT,
)
from riptable.rt_enum import CategoryMode
from riptable.rt_numpy import isnan, arange, ones
from riptable.tests.test_utils import (
    get_categorical_data_factory_method,
    get_all_categorical_data,
)
from riptable.tests.utils import LikertDecision


three_unicode = np.array(["AAPL\u2080", "AMZN\u2082", "IBM\u2081"])
three_bytes = rt.FastArray([b'a', b'b', b'c'])
three_ints = rt.FastArray([1, 2, 3])

compare_func_names = ['__ne__', '__eq__', '__ge__', '__gt__', '__le__', '__lt__']
int_success = [
    np.array([True, False, True]),  # ne
    np.array([False, True, False]),  # eq
    np.array([False, True, True]),  # ge
    np.array([False, False, True]),  # gt
    np.array([True, True, False]),  # le
    np.array([True, False, False]),  # lt
]
same_success = [
    np.array([False, False, False]),  # ne
    np.array([True, True, True]),  # eq
    np.array([True, True, True]),  # ge
    np.array([False, False, False]),  # gt
    np.array([True, True, True]),  # le
    np.array([False, False, False]),  # lt
]
diff_success = [
    np.array([True, False, True]),  # ne
    np.array([False, True, False]),  # eq
    np.array([False, True, False]),  # ge
    np.array([False, False, False]),  # gt
    np.array([True, True, True]),  # le
    np.array([True, False, True]),  # lt
]
ShowCompareInfo = False

list_bytes = [b'b', b'b', b'a', b'd', b'c']
list_unicode = ['b', 'b', 'a', 'd', 'c']
list_true_unicode = [u'b\u2082', u'b\u2082', u'a\u2082', u'd\u2082', u'c\u2082']


decision_dict = dict(zip(LikertDecision.__members__.keys(), [int(v) for v in LikertDecision.__members__.values()],))


# TODO: Replace this with assert_array_equal(), as that implements a stricter equality comparison
#       that also accounts for NA/NaN values.
def array_equal(arr1, arr2):
    subr = arr1 - arr2
    sumr = sum(subr == 0)
    result = sumr == len(arr1)
    if not result:
        print("array comparison failed", arr1, arr2)
    return result


PythonScalar = Union[int, float, str, bytes, bool]
"""Type annotation indicating which Python types are valid for use as numpy scalars."""

PythonScalarOrWildcard = Union[PythonScalar, 'ellipsis']    # Union[PythonScalar, builtins.ellipsis]
"""Type annotation indicating which Python types are valid for use as numpy scalars, plus the ellipsis (...) which can be used as a wildcard."""


# TODO: In py38+, enable the use of typing.Literal here to improve the annotation.
CompareResult = bool    # Union[bool, Literal[NotImplemented]]


def isnan_safe(x):
    try:
        return np.isnan(x)
    except TypeError:
        return False


def scalarize(x) -> Optional[Union[np.ndarray, np.generic]]:
    # Convert the value, if it's a Python scalar (but not a numpy array scalar)
    # to support comparisons between numpy.bytes_ and str.
    # Existing array scalars and ellipsis are passed through unchanged; return None
    # for all other types, including numpy arrays (since by the time this function is called
    # we should be only operating on scalars).
    if x is Ellipsis or isinstance(x, (np.ndarray, np.generic)):
        return x
    elif np.isscalar(x):
        return np.array([x])[0]
    else:
        return None


def tuple_with_wildcard_compare_func(v: tuple, w: tuple, compare_func: Callable[[Any, Any], CompareResult]) -> CompareResult:
    """Python-based implementation of the CPython comparison function for tuples, extended with wildcard support."""
    # CPython tuple comparison implementation:
    # https://github.com/python/cpython/blob/0430dfac629b4eb0e899a09b899a494aa92145f6/Objects/tupleobject.c#L674
    if not isinstance(v, tuple) or not isinstance(w, tuple):
        return NotImplemented

    vlen = len(v)
    wlen = len(w)

    # Search for the first index where items are different.
    # Note that because tuples are immutable, it's safe to reuse
    # the vlen and wlen across the comparison calls.
    common_len = builtins.min(vlen, wlen)
    i = 0
    while i < common_len:
        # Convert the component values to numpy scalars if needed.
        # This also effectively provides a check for reasonable values and assists with the
        # isnan() checks below, especially since the checks then support riptide invalids.
        v_i = scalarize(v[i])
        w_i = scalarize(w[i])
        i += 1

        # If either component is not a scalar at this point, we have a bad value
        # so stop and give an informative error message.
        if v_i is None:
            raise ValueError(f"Component at index {i} of L.H.S. is a type ('{type(v[i])}') not supported for use as a numpy scalar.")
        elif w_i is None:
            raise ValueError(f"Component at index {i} of R.H.S. is a type ('{type(w[i])}') not supported for use as a numpy scalar.")

        # Wildcard support: if either element is NaN, that takes precedence so return False;
        # otherwise, if either element is ... (ellipsis), consider the equality comparison to be True.
        # N.B. Can avoid the math.isnan() call here if we ever care to support more-general types,
        #      we'll just need to detect the bottom-type absorbing condition by doing both an
        #      equals and not-equals comparison (if they both return False, that's a NaN).
        # TODO: As of 2020-11-13, this doesn't recognize riptide integer invalids, because extracting an
        #       element of a FastArray returns a normal numpy array scalar, rather than some kind of FastArray
        #       scalar that'd recognize when it's holding the invalid value for it's dtype.
        if isnan_safe(v_i) or isnan_safe(w_i):
            return False

        if v_i is Ellipsis or w_i is Ellipsis:
            continue

        # N.B. The CPython implementation does an equality check (Py_EQ) here,
        # so we do the same thing to match.
        if not (v_i == w_i):
            break

    if i >= vlen or i >= wlen:
        # No more items to compare -- compare sizes
        return compare_func(vlen, wlen)

    # We have an item that differs.
    # Compare the final item again using the proper operator.
    return v[i] is Ellipsis or w[i] is Ellipsis or compare_func(scalarize(v[i]), scalarize(w[i]))


def reference_comparison(
    op_name: str,
    x: Union[rt.Categorical, Tuple[PythonScalarOrWildcard, ...], PythonScalarOrWildcard],
    y: Union[rt.Categorical, Tuple[PythonScalarOrWildcard, ...], PythonScalarOrWildcard]
) -> np.ndarray:
    """
    Test helper function implementing a lexicographically-ordered (tuple-style) comparison
    between a `rt.Categorical` or tuple of scalars and another `rt.Categorical` or tuple of scalars.

    When this function is used to compare a tuple of scalars with an `rt.Categorical`, a numpy-style
    broadcast is performed so the tuple is compared against each element of the `Categorical`.

    .. warning::
        This function is designed to be used as a reference/comparison when implementing unit tests.
        It emphasizes correctness above any other concern; as such, it is relatively slow and should
        not be used in production code.

    Parameters
    ----------
    op_name : str
        The name of the comparison operation to perform, e.g. ``__eq__``, ``__gt__``.
    x : rt.Categorical or tuple
    y : rt.Categorical or tuple

    Returns
    -------
    comparison_result : np.ndarray of bool

    Notes
    -----
    This function is a combination of the standard Python logic for comparing tuples and numpy logic
    for upcasting during comparisons and broadcasting.

    There is one important deviation from the standard logic:
    Python allows comparison of tuples with unequal arity/rank, e.g. ``(2, 2) > (2, 1, 0)``. We require
    tuples and `Categorical`s to have the same arity/rank; note this is an arbitrary restriction -- we could
    implement the Python logic too, but requiring the arities to be equal feels like the correct approach
    for riptable and also provides a better fit to SQL / relational algebra.

    TODO:
        * The current code does the right thing to return false for any comparisons of elements
          corresponding to the invalid/NA (0 bin) in a Categorical. However, for SQL compliance
          we need to implement additional logic to check each element of each tuple for whether
          it's considered a riptable invalid/NA value -- if so, comparison against that component
          always returns False. This logic needs to respect the order in which the tuple components
          are compared -- e.g. (3, null) > (2, 10) should be True but (2, null) > (2, 10) should be
          False because the 0th components are equal, so we move to the 1st components and the
          comparison against null always returns null (which means the comparison result is False).
        * Current implementation assumes Categorical instances are the "standard" .singlekey or
          .multikey variety; the current comparison implementation in Categorical also handles (some)
          cases for the "enum" and "dict"-style Categoricals. This function (below) is meant to be a
          slow-but-accurate reference implementation, so it should also support those Categorical types.
        * Need to support comparison of a Categorical against a compatible tuple of (possibly a mixture of)
          array/scalar/ellipsis.
          * For completeness, this should probably also include 1-ary Categoricals (to be allowed in the tuple),
            in case someone wants to write e.g. my_multikey_cat == (my_singlekey_cat_a, my_enum_cat_b).
        * For the real implementation of this method within Categorical, we may want to have a boolean
          flag indicating whether integer invalids are recognized as such (or treated as if integers
          have no invalids). It's not much more work to implement both versions and it'll allow
          calling code to be explicit about which behavior it's expecting.
          * Provide another bool flag to allow callers to specify what value comparisons with nulls
            should return. Comparison operators would specify False, for example, to provide the
            standard behavior with NaNs (where any comparison with a NaN returns false); however,
            some other callers might want to do something like a wildcard comparison -- for example,
            if we have a Categorical with 4-valued integer versions (major, minor, build, revision)
            and we want to do a comparison like my_version_cat >= (2, inv, 5291, inv). We could fill
            in zeros for those wildcards, but it'd be more general to do this using the proposed flag
            so the behavior always works correctly even for signed integers, Date, etc.
        * Simplify the implementation a bit using the 'multipledispatch' or 'plum' library (or some
          other library providing support for defining multiple-dispatch methods)?

    Additional notes:
        * string-typed categories are allowed to be compared across Unicode / bytes.
          The scalar string will be converted (if needed) to the type of the category array.
        * ordering comparisons (inequality relations) may only be performed when the Categorical
          is an ordered Categorical; otherwise, an error is expected to be raised.
        * for integer-valued Categoricals, we allow comparison/indexing with the integers as strings.
          It's unclear what the use case is for this logic -- we should document that behavior.
    """
    def cat_to_tuple_list(cat: rt.Categorical) -> Sequence[tuple]:
        return list(zip(*cat.category_dict.values()))

    def compat_string_encode(cat_values_dtype: np.dtype, x: PythonScalar, component_idx: int) -> PythonScalar:
        # If this scalar is a string, convert it to the type compatible with the category values.
        # This is normally not allowed in numpy (or riptable) -- comparing arrays/scalars of different encoding types
        # causes an error to be raised -- but users find it useful when working with Categorical so we allow it.
        if isinstance(x, (bytes, str)):
            if cat_values_dtype.char == 'S':
                if not isinstance(x, bytes):
                    try:
                        return x.encode('ascii')
                    except UnicodeEncodeError:
                        raise TypeError(f"Unable to convert Unicode string to ASCII string (bytes) for the component at index {component_idx}.")
            elif cat_values_dtype.char == 'U':
                if not isinstance(x, str):
                    return x.decode()

        # Fallthrough -- if we didn't need to change the scalar, just pass it back.
        return x

    compare_func = getattr(operator, op_name)

    if isinstance(x, rt.Categorical):
        if isinstance(y, rt.Categorical):
            # The Categoricals must have the same length (number of elements).
            if len(x) != len(y):
                raise ValueError("Cannot compare Categoricals with different lengths.")

            x_cat_dict = x.category_dict
            y_cat_dict = y.category_dict

            # The categoricals must have the same arity/rank (i.e. same number of category arrays).
            # They may have a different number of category values though.
            if len(x_cat_dict) != len(y_cat_dict):
                raise ValueError(f"Categoricals of unequal arity/rank cannot be compared. (x.rank={len(x_cat_dict)}, y.rank={len(y_cat_dict)})")

            # Convert the category values in 'x' and 'y' to a list of tuples of numpy scalars.
            # This gives us the correct combination of comparison logic for the next step.
            x_cat_tuples = cat_to_tuple_list(x)
            y_cat_tuples = cat_to_tuple_list(y)
            assert builtins.all(map(lambda z: np.isscalar(z[0]), x_cat_tuples))

            # Compare each category (key-tuple) from 'x' to those from 'y' to
            # produce an m x n matrix of comparison results. Note these are only
            # for the category values -- we'll need to fancy-index into this
            # array to get the actual result array we want to return.
            cat_comparison_results = rt.zeros((len(x_cat_tuples) + x.base_index, len(y_cat_tuples) + y.base_index), dtype=np.bool)
            cat_comparison_results_view = cat_comparison_results[x.base_index:, y.base_index:]
            for i in range(len(x_cat_tuples)):
                for j in range(len(y_cat_tuples)):
                    # Use the regular compare_func here instead of tuple_with_wildcard_compare_func() because we don't
                    # support wildcard values inside Categoricals.
                    cat_comparison_results_view[i, j] = compare_func(x_cat_tuples[i], y_cat_tuples[j])

            # Create a mask indicating which elements in the fancy index (created below) correspond to _valid_
            # elements (i.e. not filtered/invalid) in one or both of the Categoricals.
            # As of riptable 1.0.25, rt.isnan() doesn't work as expected for Categoricals so we need to
            # do that manually.
            x_isvalid = True if x.base_index == 0 else x._fa != 0
            y_isvalid = True if y.base_index == 0 else y._fa != 0
            result_valid_mask = np.logical_and(x_isvalid, y_isvalid)

            # Combine the underlying data arrays for the Categoricals -- which individually can be used
            # as a fancy index (as long as the .base_offset is accounted for) -- into a single fancy index
            # that can be used to index into a flattened view of the comparison results to produce the
            # final comparison output (that'll have the correct shape to match the input Categoricals).
            # NOTE: The implementation here could in theory be simplified to index into the 2D comparison results
            #       array like ``cat_comparison_results[arr1, arr2]`` but riptable (as of 1.0.24) punts to
            #       numpy when rt.mbget() is called on multi-dimensional arrays. If that's ever fixed,
            #       make the simplification above since it'll remove two intermediate array allocations
            #       by pushing the multiplication and addition operations down to the C++ implementation.
            # NOTE: The np.int32() usage is important here -- the underlying categorical arrays are likely
            #       int8/int16/int32, so if we don't force them to be upcasted to a larger type, the index
            #       calculation silently overflows and the results later on will be incorrect.
            result_fa = (np.int32(len(y_cat_tuples) + y.base_index) * x._fa) + y._fa
            assert np.all(result_fa >= 0), "Invalid/impossible array indices."

            # Use the constructed fancy index to build the array of comparison results.
            # Because riptide_cpp / riptable (as of 1.0.24) don't propagate integer invalids through
            # operations like multiplication and addition, the fancy index we constructed above
            # could have some valid-looking indices created from one or more *invalid* indices.
            # To handle that, we use np.copyto() to overwrite the corresponding result values
            # with False to mimic the behavior of how comparisons behave with NaNs in IEEE754.
            comparison_results = cat_comparison_results.ravel()[result_fa]
            np.copyto(comparison_results, False, casting='no', where=result_valid_mask)

            return comparison_results

        elif isinstance(y, tuple):
            x_cat_dict = x.category_dict

            # The tuple must have the same number of components as the Categorical has category columns.
            # This also catches empty tuples, since it's not valid to have a 0-ary Categorical.
            if len(x_cat_dict) != len(y):
                raise ValueError(f"Cannot compare a Categorical on {len(x_cat_dict)} columns against a tuple of {len(y)} components.")

            # Create the array that'll hold our category-level comparison results.
            # This is done in a way that the array shape is compatible with the Categorical's underlying array,
            # which lets us produce the final output (with the same shape as the 'x' Categorical) by fancy-indexing it.
            # PERF: For the 'real', optimized implementation of this method, try to avoid calling .nunique() -- it gives
            #       us the value we need here in all cases, but is relatively slow. Or, we could just optimize the
            #       implementation of nunique() so it's fast for non-.isenum cases.
            result_invalid_offset = 0 if x.base_index is None else x.base_index
            cat_comparison_results = np.zeros(x.nunique() + result_invalid_offset, dtype=np.bool)

            # If there's an invalid bin, we want to preset it to False; given the way we build up the results below,
            # setting the comparison result for the invalid bin here ensures comparisons against invalids always return False.
            cat_comparison_results[:result_invalid_offset] = False
            cat_comparison_results_view = cat_comparison_results[result_invalid_offset:]

            # "enum"-style Categoricals have somewhat different characteristics than "normal" Categoricals;
            # to simplify the implementation the .isenum case is handled separately from the other Categorical types.
            # N.B. It seems like x.isenum should imply x.ordered, but that's not currently the case due to the
            #   mixture of Grouping and Categories used within Categorical. If that's ever cleaned up, this check could
            #   drop the additional check for .isenum.
            if x.isenum:
                # "enum"-style Categoricals should always be considered 'ordered'.
                # They also require all operations -- incl. comparisons -- to operate _only_ over the
                # defined cases of the enum; any other inputs must cause an error to be raised.
                raise NotImplementedError

            else:   # assumes .issinglekey or .ismultikey
                # Can't perform inequality relational operations on unordered Categoricals.
                if not x.ordered and op_name not in ('__eq__', '__ne__'):
                    raise ValueError(f"Cannot make accurate comparison with {op_name} on unordered Categorical.")

                comparison_in_progress = np.ones_like(cat_comparison_results_view)

                # Iterate over each component of the tuple and the corresponding category value array from the Categorical,
                # performing a comparison between them and updating the results.
                for (key_name, cat_key_values), (component_idx, tup_component) in zip(x_cat_dict.items(), enumerate(y)):
                    # Short-circuit if no category tuples are still being compared.
                    if not np.any(comparison_in_progress):
                        break

                    # Comparisons with NaN/NA/invalid always return False (according to SQL semantics and IEEE754).
                    # TODO: Whenever FastArray.__getitem__ returns riptable scalars (which support integer NA values),
                    #       check the tuple component for NaN/NA here too in case we're given one of those.
                    # TODO: Fix this -- it won't give the correct behavior for categories which've already had their
                    #       comparison "decided". E.g. we expect something like (3, nan) > (2, 1.5), because the first
                    #       components are different and decide the outcome so we don't consider the second elements.
                    #       Maybe just calculate the rt.isnotnan() here since it's relevant to all cases below, and rely
                    #       on them to combine the values into the results.
                    # TEMP: Work around rt.isnotnan() not working for string arrays (as of 2020-11-17)
                    category_valid_components = True if cat_key_values.dtype.char in 'SU' else rt.isnotnan(cat_key_values)

                    if np.isscalar(tup_component):
                        if isinstance(tup_component, (bytes, str)):
                            # TODO: Is this logic still necessary? If so, can it be improved at all, e.g. to get rid of the CategoryMode
                            #       check (since we'd eventually like to get rid of that)?
                            #       Source: Categorical._categorical_compare_check()
                            if x.category_mode != CategoryMode.StringArray and not x.isenum:
                                # try to convert to int
                                # this happens when c=Cat([1,2,3]); c['2']
                                try:
                                    # extract float or integer
                                    fnum = float(tup_component)
                                    tup_component = int(tup_component) if round(fnum) == fnum else fnum
                                except Exception as ex:
                                    # TODO: Include 'component_idx' in this error message to provide more context for users.
                                    raise TypeError(f"Comparisons to single strings can only be made to categoricals in StringArray mode - not {x.category_mode.name} mode.  Error {ex}")

                            # For string-typed scalars, provide a convenience to users by allowing (and adjusting here)
                            # different string encodings to be used.
                            tup_component = compat_string_encode(cat_key_values.dtype, tup_component, component_idx)

                        # Determine for which categories the tuple component matches the corresponding category component.
                        components_eq = np.logical_and(category_valid_components, operator.eq(cat_key_values, tup_component))

                        # Compare the category values vs. the scalar.
                        component_cmp_results = compare_func(cat_key_values, tup_component)

                        # For any elements which were/are still in-progress and the tuple components are _not_ equal,
                        # we've reached their stopping point. Store the comparison results for these elements then
                        # remove them from the mask indicating the still-in-progress elements.
                        np.copyto(
                            cat_comparison_results_view,
                            np.logical_and(category_valid_components, component_cmp_results),
                            where=np.logical_and(comparison_in_progress, np.logical_not(components_eq)))
                        comparison_in_progress = np.logical_and(comparison_in_progress, components_eq)

                    elif isinstance(tup_component, np.ndarray):
                        # TODO: Validate array shape and type
                        raise NotImplementedError("TODO: Implement support for comparing a Categorical with a shape- and type-compatible array.")

                    elif tup_component is Ellipsis:
                        # Comparison with ellipsis (wildcard) always returns True (except for NaN-comparison, that takes precedence).
                        # Update the results for any categories that have a NaN/NA/invalid component.
                        if not np.all(category_valid_components):
                            # The default value in the per-category results array is False, so all we need to do here
                            # is remove the categories with invalid components at this index from the mask of categories
                            # still being compared.
                            comparison_in_progress = np.logical_and(comparison_in_progress, category_valid_components)

                    else:
                        raise ValueError(f"The component of 'y' at index {component_idx} is not a scalar, np.ndarray, or ellipsis.")

                # If any categories are still marked as in-progress, it should be because the comparison op
                # includes equality (__eq__, __ge__, __le__) and all components of the category are equal
                # to the tuple. The exception to this rule -- where it can be true for other operations --
                # is when all components of the tuple are wildcards.
                if np.any(comparison_in_progress):
                    assert op_name in ('__eq__', '__ge__', '__le__') or all(map(lambda x: x is Ellipsis, y))
                    cat_comparison_results_view[comparison_in_progress] = True

            # Use the underlying data array for 'x' as a fancy index into the category comparison results
            # to produce a results array that matches up to 'x'.
            comparison_results = cat_comparison_results[x._fa]

            return comparison_results

        # All comparisons/checks below this only support 1-ary Categoricals (.issinglekey, .isenum).
        # Check if the Categorical we have is a multi-key; if it is, this is a good place to short-circuit
        # and raise the exception since it means we don't need to do it in each branch below.
        elif x.ismultikey:
            raise ValueError(
                "Categoricals can only be compared to other Categoricals, scalars (for 1-ary Cats), or same-arity tuples of scalar/array/ellipsis.")

        elif y is Ellipsis:
            # Compare ... (ellipsis) as a wildcard against a 1-ary Cat (.issinglekey, .isenum).
            # All this needs to do is call + return .isnotnan() on the Categorical,
            # since any comparison against a wildcard returns True except for NaN/NA values.
            # return x.isnotnan()
            return reference_comparison(op_name, x, (y,))

        elif np.isscalar(y):
            # Compare scalar against a 1-ary Cat (.issinglekey, .isenum).
            return reference_comparison(op_name, x, (y,))

        elif isinstance(y, np.ndarray):
            # Compare array against a 1-ary Cat (.issinglekey, .isenum).
            return reference_comparison(op_name, x, (y,))

        else:
            # This is some unsupported type -- raise an exception.
            raise TypeError(f"Support for the type '{type(y)}' is not implemented.")

    elif isinstance(y, rt.Categorical):
        # These cases are implemented by swapping the order of the operands and using the inverse operation
        # (for inequalities; for eq/ne operand order doesn't matter), then making a recursive call.
        # This approach minimizes the complexity of the implementation by avoiding (nearly-)duplicated code.
        inverse_op_names = {'__lt__': '__ge__', '__le__': '__gt__', '__gt__': '__le__', '__ge__': '__lt__'}
        swapped_op_name = inverse_op_names.get(op_name, op_name)
        return reference_comparison(swapped_op_name, y, x)

    else:
        raise TypeError("At least one of the two comparands must be a Categorical.")


class TestCategoricalCompare:
    """Tests for comparisons with/against `Categorical` instances."""

    def test_compare_check(self):
        '''
        Test comparison between two 'equal' categoricals with different underlying arrays.
        '''
        compare_ops = {
            '__ne__': [False, False, False, False, False, False],
            '__eq__': [True, True, True, True, True, True],
            '__ge__': [True, True, True, True, True, True],
            '__gt__': [False, False, False, False, False, False],
            '__le__': [True, True, True, True, True, True],
            '__lt__': [False, False, False, False, False, False],
        }
        c = rt.Categorical(['b', 'b', 'b', 'a', 'b', 'b'], ['a', 'b', 'c'])
        d = rt.Categorical(['b', 'b', 'b', 'a', 'b', 'b'], ['a', 'b'])
        for name, correct in compare_ops.items():
            func = c.__getattribute__(name)
            result = func(d)
            is_correct = bool(np.all(result == correct))
            assert is_correct, f"Compare operation betweeen two equal categoricals did not return the correct result."

    @pytest.mark.parametrize("op_name", ['__ne__', '__eq__', '__ge__', '__gt__', '__le__', '__lt__'])
    def test_compare_return_type(self, op_name: str):
        '''
        Test comparison operations with single strings to make sure FastArray of boolean is returned.
        '''
        c = Categorical(['b', 'b', 'b', 'a', 'b', 'b'], ['a', 'b'])
        scalars = ['a', 'c']
        for s in scalars:
            func = c.__getattribute__(op_name)
            result = func(s)
            assert isinstance(result, FastArray), f"comparison {op_name} with input {s} did not return FastArray"
            assert result.dtype.char == '?', f"comparison {op_name} with input {s} did not return boolean"

    def test_compare_different_modes(self):
        c1 = Categorical(['b', 'b', 'b', 'a', 'b', 'b'], ['a', 'b'])
        c2 = Categorical([0, 1], {0: 'a', 1: 'b'})
        with pytest.raises(TypeError):
            _ = c1 == c2

    def test_compare_conflicting_dicts(self):
        c1 = Categorical([0, 1], {0: 'a', 1: 'b'})
        c2 = Categorical([0, 1], {1: 'a', 0: 'b'})
        with pytest.raises(ValueError):
            _ = c1 == c2

    def test_compare_safe_dicts(self):
        c1 = Categorical([0, 1], {0: 'a', 1: 'b'})
        c2 = Categorical([2, 1], {2: 'c', 1: 'b'})
        correct = FastArray([False, True])
        result = c1 == c2
        assert_array_equal(correct, result)

    @pytest.mark.parametrize("op_name", compare_func_names)
    @pytest.mark.parametrize("compatible_strs", [False, True])
    def test_compare_multikey_cat_vs_multikey_cat(self, op_name: str, compatible_strs: bool):
        """Test comparison logic for two multikey Categoricals."""

        # TODO: Implement checks to verify an error is raised when:
        #   * the Categoricals are different lengths
        #   * the Categoricals have different rank (number of category columns)
        #   * incompatible/uncomparable category arrays (e.g. categorical 'A's 2nd category column
        #     is a string array but categorical 'B's 2nd category column is an integer or Date).

        # Create two compatible multikey Categoricals to test the comparison logic.
        cat_a_filter = rt.FA(
            [True, True, True, True, True, True, False, True, True, True, False, True, True, False, True, False, True, True, True, True], dtype=np.bool)
        cat_a = rt.Categorical([
            rt.FA([8, 6, 8, 0, 9, 1, 1, 0, 1, 5, 4, 6, 5, 0, 3, 5, 2, 3, 4, 9], dtype=np.int32),
            rt.FA([358, 895, 358, 895, 0, -179, -358, -716, 537, 537, 179, 716, -537, -895, -716, -179, -716, 179, 537, 358], dtype=np.int16),
            rt.FA([
                b'45fe446c69', b'f24b80397c', b'45fe446c69', b'abc60647fd', b'56dee0a8b0',
                b'7afe8c5d3f', b'171674b99c', b'2f0438b10a', b'7d780022b8', b'c34742169f',
                b'e954c312b2', b'eb43899a35', b'a89a2b870d', b'd2ac2b5359', b'2f078bace8',
                b'40fe99d050', b'9752e063e9', b'4a66f69fb5', b'6907359a83', b'dd999ad979'], dtype="S10"),
            # TODO: Also add an rt.Date array here
        ], filter=cat_a_filter)

        cat_b_filter = rt.FA(
            [True, True, True, True, False, True, True, False, True, False, False, True, True, True, False, True, True, True, True, False], dtype=np.bool)
        cat_b = rt.Categorical([
            rt.FA([6, 6, 4, 9, 8, 5, 9, 8, 3, 6, 4, 4, 5, 3, 2, 4, 0, 1, 4, 2], dtype=np.int32),
            rt.FA([179, 0, 537, 358, 358, 179, 179, 716, 895, 716, 179, 895, 537, 716, 895, 0, 0, 716, 537, 0], dtype=np.uint16),
            rt.FA([
                '3c62ed9ba7', '4d159968fc', '6907359a83', '0426d59978', '45fe446c69',
                '6be335debf', '62e6c9eee4', 'eae9f57cdc', '663d6aeea1', '23553e0997',
                '8c0ee4604f', '89db4cc641', 'a89a2b870d', 'df35217de5', '422c78fd7f',
                'f4d37c22a4', '5caacb68c0', '1f2601e91c', '6907359a83', '8685e75d78'], dtype="U10").astype('S' if compatible_strs else 'U'),
            # TODO: Also add an rt.Date array here
        ], filter=cat_b_filter)

        # Compare using the strict, Python-tuple-based reference implementation.
        try:
            reference_result = reference_comparison(op_name, cat_a, cat_b)
        except BaseException as reference_exc:
            print(f"Reference func exception: {reference_exc}")
            reference_result = reference_exc

        # Compare the two Categoricals using the specified comparison method
        # on Categorical itself.
        cat_compare_func = getattr(cat_a, op_name)
        try:
            cat_compare_result = cat_compare_func(cat_b)
        except BaseException as cat_compare_exc:
            print(f"Implementation func exception: {cat_compare_exc}")
            cat_compare_result = cat_compare_exc

        # Are the results the same?
        # If an exception was raised, we only check that the same type was raised in both cases;
        # we don't inspect the error messages (they are useful for users, but it's almost certainly the
        # case we'll have different wording between these two implementations, so it's not useful
        # to this test to require the messages to be the same).
        if isinstance(reference_result, BaseException):
            # Exception types must match exactly.
            assert type(reference_result) == type(cat_compare_result)
        else:
            assert_array_equal(reference_result, cat_compare_result)

    @pytest.mark.parametrize("op_name", compare_func_names)
    @pytest.mark.parametrize("tup", [
        pytest.param(
            (6, 179, b'3c62ed9ba7'),
            id="first two components have at least one exact row match, last component does not"
        ),
        pytest.param(
            (4, 537, b'6907359a83'),
            id="exact match to 2nd-to-last row"
        ),
        # This next case is an exact match between the values of the tuple and the 2nd-to-last
        # row of the Categorical data EXCEPT for the fact that the strings in the last positions
        # have different types (ascii vs. Unicode). Neither numpy not riptide allow that comparison
        # and should give a TypeError when comparing two such arrays -- this just tests that the
        # comparison implementation respects that and doesn't e.g. attempt to downconvert the
        # Unicode strings to ASCII, since riptide sometimes does that when it's feasible for better perf.
        pytest.param(
            (4, 537, '6907359a83'),
            id="exact match to 2nd-to-last row except for different string types"
        ),
        pytest.param(
            (1, -358, b'171674b99c'),
            id="compare tuple against filtered Categorical row"
        ),
        (2, 0, b'8685e75d78'),
        pytest.param(
            (3, 179, ...),
            id="tuple with wildcard in last position"
        ),
        pytest.param(
            (3, ..., b'171674b99c'),
            id="tuple with wildcard somewhere besides first/last"
        ),
        pytest.param(
            (..., 179, ...),
            id="tuple with wildcards at ends"
        ),
        pytest.param(
            (..., ..., ...),
            id="tuple with all wildcards"
        )
    ])
    @pytest.mark.parametrize("ordered_cat", [False, True])
    def test_compare_multikey_cat_vs_tuple(self, op_name: str, tup: tuple, ordered_cat: bool):
        """Test comparison logic for two multikey Categoricals."""

        # TODO: Implement checks to verify an error is raised when:
        #   * the Categoricals are different lengths
        #   * the Categoricals have different rank (number of category columns)
        #   * incompatible/uncomparable category arrays (e.g. categorical 'A's 2nd category column
        #     is a string array but categorical 'B's 2nd category column is an integer or Date).

        # Create a multikey Categorical to test the comparison logic.
        cat_a_filter = rt.FA(
            [True, True, True, True, True, True, False, True, True, True, False, True, True, False, True, False, True, True, True, True], dtype=np.bool)
        cat_a = rt.Categorical([
            rt.FA([8, 6, 8, 0, 9, 1, 1, 0, 1, 5, 4, 6, 5, 0, 3, 5, 2, 3, 4, 9], dtype=np.int32),
            rt.FA([358, 895, 358, 895, 0, -179, -358, -716, 537, 537, 179, 716, -537, -895, -716, -179, -716, 179, 537, 358], dtype=np.int16),
            rt.FA([
                b'45fe446c69', b'f24b80397c', b'45fe446c69', b'abc60647fd', b'56dee0a8b0',
                b'7afe8c5d3f', b'171674b99c', b'2f0438b10a', b'7d780022b8', b'c34742169f',
                b'e954c312b2', b'eb43899a35', b'a89a2b870d', b'd2ac2b5359', b'2f078bace8',
                b'40fe99d050', b'9752e063e9', b'4a66f69fb5', b'6907359a83', b'dd999ad979'], dtype="S10"),
            # TODO: Also add an rt.Date array here
        ], filter=cat_a_filter, ordered=ordered_cat)

        # Compare using the strict, Python-tuple-based reference implementation.
        try:
            reference_result = reference_comparison(op_name, cat_a, tup)
        except BaseException as reference_exc:
            print(f"Reference func exception: {reference_exc}")
            reference_result = reference_exc

        # Compare the two Categoricals using the specified comparison method
        # on Categorical itself.
        cat_compare_func = getattr(cat_a, op_name)
        try:
            cat_compare_result = cat_compare_func(tup)
        except BaseException as cat_compare_exc:
            print(f"Implementation func exception: {cat_compare_exc}")
            cat_compare_result = cat_compare_exc

        # Are the results the same?
        # If an exception was raised, we only check that the same type was raised in both cases;
        # we don't inspect the error messages (they are useful for users, but it's almost certainly the
        # case we'll have different wording between these two implementations, so it's not useful
        # to this test to require the messages to be the same).
        if isinstance(reference_result, BaseException):
            # Exception types must match exactly.
            assert type(reference_result) == type(cat_compare_result)
        else:
            assert_array_equal(reference_result, cat_compare_result)

    def test_tuple_compare_error(self):
        c = rt.Categorical([rt.FastArray(['a', 'b', 'c', 'a']), rt.FastArray([1, 2, 3, 1])])
        with pytest.raises(ValueError):
            _ = c == ('a', 'b', 'c')

    def test_bytes_compare_multikey(self):
        c = rt.Categorical([np.array(['a', 'b', 'c', 'a']), rt.FastArray([1, 2, 3, 1])], unicode=True)
        cols = c.category_dict
        bytescol = list(cols.values())[0]
        assert bytescol.dtype.char == 'U'
        result = c == (b'a', 1)
        expected = rt.FastArray([True, False, False, True])
        assert_array_equal(expected, result)

    # TODO pytest parameterize `compare_func_names`
    # TODO: Also split the tests within this function into multiple separate tests
    def test_all_compare_tests(self):
        # with scalar
        # cat(unicode)
        i = 2
        c1 = Categorical(three_ints)
        if ShowCompareInfo:
            print("Categorical:", c1)
        if ShowCompareInfo:
            print("Compare unicode to int scalar: 2")
        self.compare_cat_test(c1, compare_func_names, int_success, i)

        # cat(unicode) / unicode, unicode list
        i = "AMZN\u2082"
        c3 = Categorical(three_unicode)
        if ShowCompareInfo:
            print("Categorical:", c3)
        if ShowCompareInfo:
            print("Compare unicode cat to unicode string")
        self.compare_cat_test(c3, compare_func_names, int_success, i)
        if ShowCompareInfo:
            print("Compare to list of unicode string")
        self.compare_cat_test(c3, compare_func_names, int_success, [i])
        if ShowCompareInfo:
            print("Compare to a numpy array of unicode string")
        self.compare_cat_test(c3, compare_func_names, int_success, np.array([i]))

        # cat(bytes) / bytes, bytes list
        i = b'b'
        c4 = Categorical(three_bytes)
        if ShowCompareInfo:
            print("Categorical:", c4)
        if ShowCompareInfo:
            print("Compare bytes cat to bytestring")
        self.compare_cat_test(c4, compare_func_names, int_success, i)
        if ShowCompareInfo:
            print("Compare to bytestring in list")
        self.compare_cat_test(c4, compare_func_names, int_success, [i])
        if ShowCompareInfo:
            print("Compare to bytestring in numpy array")
        self.compare_cat_test(c4, compare_func_names, int_success, np.array([i]))

        # cat(bytes) / unicode, unicode list
        i = "b"
        c5 = Categorical(three_bytes)
        if ShowCompareInfo:
            print("Categorical:", c5)
        if ShowCompareInfo:
            print("Compare bytes cat to unicode string")
        self.compare_cat_test(c5, compare_func_names, int_success, i)
        if ShowCompareInfo:
            print("Compare to unicode string in list")
        self.compare_cat_test(c5, compare_func_names, int_success, [i])
        if ShowCompareInfo:
            print("Compare to unicode string in numpy array")
        self.compare_cat_test(c5, compare_func_names, int_success, np.array([i]))

        # equal categoricals (same dictionary)
        # cat(bytes) / cat(bytes)
        if ShowCompareInfo:
            print("Compare two equal categoricals:")
        if ShowCompareInfo:
            print("Both from byte lists:")
        c1 = Categorical(three_bytes)
        c2 = Categorical(three_bytes)
        if ShowCompareInfo:
            print("cat1:", c1)
        if ShowCompareInfo:
            print("cat2:", c2)
        self.compare_cat_test(c1, compare_func_names, same_success, c2)

        # cat(unicode) / cat(unicode)
        if ShowCompareInfo:
            print("Both from unicode lists:")
        c1 = Categorical(three_unicode)
        c2 = Categorical(three_unicode)
        if ShowCompareInfo:
            print("cat1:", c1)
        if ShowCompareInfo:
            print("cat2:", c2)
        self.compare_cat_test(c1, compare_func_names, same_success, c2)

        # cat(unicode) / cat(bytes)
        if ShowCompareInfo:
            print("unicode/bytes list")
        c1 = Categorical(["a", "b", "c"])
        c2 = Categorical(three_bytes)
        if ShowCompareInfo:
            print("cat1:", c1)
        if ShowCompareInfo:
            print("cat2:", c2)
        self.compare_cat_test(c1, compare_func_names, same_success, c2)

        # unequal categoricals (same dictionary)
        # cat(bytes) / cat(bytes)
        if ShowCompareInfo:
            print("Compare two unequal categoricals (same dict):")
        if ShowCompareInfo:
            print("both bytes")
        c1 = Categorical([0, 1, 0], three_bytes)
        c2 = Categorical([2, 1, 2], three_bytes)
        if ShowCompareInfo:
            print("cat1:", c1)
        if ShowCompareInfo:
            print("cat2:", c2)
        self.compare_cat_test(c1, compare_func_names, diff_success, c2)

        # cat(unicode) / cat(unicode)
        if ShowCompareInfo:
            print("both unicode")
        c1 = Categorical([0, 1, 0], three_unicode)
        c2 = Categorical([2, 1, 2], three_unicode)
        if ShowCompareInfo:
            print("cat1:", c1)
        if ShowCompareInfo:
            print("cat2:", c2)
        self.compare_cat_test(c1, compare_func_names, diff_success, c2)

        ## cat(bytes) / int list (matching)
        # if ShowCompareInfo: print("Compare categorical to matching int list")
        # if ShowCompareInfo: print("bytes")
        # i = [1,2,3]
        # c1 = Categorical(three_bytes)
        # self.compare_cat_test(c1,compare_func_names,same_success,i)
        ## cat(unicode) / int list (matching)
        # if ShowCompareInfo: print("unicode")
        # c1 = Categorical(three_unicode)
        # self.compare_cat_test(c1,compare_func_names,same_success,i)

        ## cat(bytes) / int list (non-matching)
        # if ShowCompareInfo: print("Compare categorical to non-matching int list")
        # if ShowCompareInfo: print("bytes")
        # i = [3,2,1]
        # c1 = Categorical(three_bytes)
        # self.compare_cat_test(c1,compare_func_names,int_success,i)
        ## cat(unicode) / int list(non-matching)
        # if ShowCompareInfo: print("unicode")
        # c1 = Categorical(three_unicode)
        # self.compare_cat_test(c1,compare_func_names,int_success,i)

    # TODO move this to testing utils
    def compare_cat_test(self, cat, compare_func_names, success_bools, i):
        for fname, success in zip(compare_func_names, success_bools):
            func = getattr(cat, fname)
            result = func(i)
            assert np.all(result == success), f'fail on {fname} {cat} {i}'
            if ShowCompareInfo:
                if np.all(result == success):
                    message = "succeeded"
                else:
                    message = "failed"
                print(fname, message)


