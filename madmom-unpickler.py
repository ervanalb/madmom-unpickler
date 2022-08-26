from madmom.ml.nn import NeuralNetworkEnsemble, NeuralNetwork, average_predictions
from madmom.ml.nn.activations import tanh, sigmoid
from madmom.ml.nn.layers import LSTMLayer, FeedForwardLayer, Gate, Cell
from madmom.processors import ParallelProcessor
import base64
import madmom.models
import numpy as np

# Sentinel value indicating object should be omitted from kwargs
omit = object()

# Builtin <class 'function'> (can't figure out how to import this)
function = type(lambda: None)


def _getattr_if(attr, include_if):
    def getter(obj):
        v = getattr(obj, attr)
        return v if include_if(v) else omit

    return getter


_not_none = lambda v: v is not None


def code_gen(obj, compress_arrays=True, allow_missing=False):
    """A function which takes in an object
    (typically a madmom model but other Python objects may als work)
    and returns a string of Python code that can be used to reconstruct it.

    * `obj` is the object to serialize into Python code
    * `compress_arrays` sets whether numpy arrays are represented
      as lists of human-readable decimal numbers (False)
      or as strings of bytes using base64 encoding (True)
    * `allow_missing` sets whether serialization will crash
      on the first unrecognized object (False)
      or will continue, using the string MISSING(type) in the output (True)
    """

    # Serialization is based around generator functions
    # (typically functions named `gen_*()`)
    # These functions take in a Python object
    # and return a string containing Python code to construct that object.
    # They may recursively call the generic `gen()` function
    # to serialize nested objects.

    def gen_factory(*args, **kwargs):
        # This factory function produces a common kind of gen_*() function.
        # The returned gen_*() function will call the object's constructor
        # (whose name is inferred from type(obj))
        # with a set of positional arguments and keyword arguments
        # matching those passed to the factory function.

        # Example usage:
        # gen_MyClass = gen_factory("a", "b", c="c")
        # code = gen_MyClass(my_object)
        # This will produce something like:
        # 'MyClass(5, 6, c=7)'
        # assuming my_object had attributes a, b, and c
        # with values 5, 6, and 7.

        # Each passed-in parameter can be a string
        # (in which case the relevant value will be looked up using getattr(obj, str))
        # or a callable lookup function
        # (in which case the relevant value will the result of calling the lookup function
        # on the object)

        # Keyword arguments are optional,
        # and the generated code will not emit a keyword parameter
        # if the lookup function returns the special sentinel value `omit`.

        # Example usage:
        # gen_MyClass = gen_factory("a", "b", c=lambda obj: obj.c if obj.c is not None else omit)
        # code = gen_MyClass(my_object)
        # This will produce:
        # 'MyClass(5, 6, c=7)'
        # if my_object.c was 7, or
        # 'MyClass(5, 6)'
        # if my_object.c was None

        # Note: The above can also be written using convenience functions:
        # gen_factory("a", "b", c=_getattr_if("c", _not_none))

        def ensure_lookup_fn(descriptor):
            # If "descriptor" is a string, return a function that does getattr(descriptor) on an object.
            # If "descriptor" is already a function, return it unchanged

            if isinstance(descriptor, str):
                return lambda obj: getattr(obj, descriptor)
            elif callable(descriptor):
                return descriptor
            else:
                assert False, "descriptor must be string or function"

        args = [ensure_lookup_fn(arg) for arg in args]
        kwargs = {k: ensure_lookup_fn(v) for (k, v) in kwargs.items()}

        def code_gen_fn(obj):
            args_string_list = [gen(arg_lkup(obj)) for arg_lkup in args]
            kwargs_obj_dict = {
                arg_name: arg_lkup(obj) for (arg_name, arg_lkup) in kwargs.items()
            }
            kwargs_obj_dict = {
                k: v for (k, v) in kwargs_obj_dict.items() if v is not omit
            }
            kwargs_string_list = [
                f"{arg_name}={gen(arg_val)}"
                for (arg_name, arg_val) in kwargs_obj_dict.items()
            ]

            args_string = ",".join(args_string_list + kwargs_string_list)

            return f"{obj.__class__.__name__}({args_string})"

        return code_gen_fn

    # gen function for a generic sequence (probably needs some additional braces or brackets)
    def gen_seq(seq):
        return ",".join(gen(elem) for elem in seq)

    # gen function for a NeuralNetworkEnsemble
    # This one is a little weird because rather than using named attributes,
    # the constructor args get stored in a two-element list called obj.processors
    gen_NeuralNetworkEnsemble = gen_factory(
        lambda obj: obj.processors[0],  # 'networks' parameter
        ensemble_fn=lambda obj: obj.processors[1]
        if obj.processors[1] is not average_predictions
        else omit,
    )

    # Unchecked gen function for a ParallelProcessor
    # Objects with num_threads > 1 are not supported.
    # This function does not check whether the given object has num_threads > 1
    gen_ParallelProcessor_unchecked = gen_factory("processors")

    # gen function for a ParallelProcessor
    # Objects with num_threads > 1 are not supported
    # and will throw an AssertionError
    def gen_ParallelProcessor(obj):
        assert obj.map is map  # True for num_threads == 1
        return gen_ParallelProcessor_unchecked(obj)

    gen_NeuralNetwork = gen_factory("layers")

    gen_LSTMLayer = gen_factory(
        "input_gate",
        "forget_gate",
        "cell",
        "output_gate",
        activation_fn=_getattr_if("activation_fn", lambda v: v is not tanh),
        cell_init=_getattr_if("cell_init", lambda v: any(v)),
    )

    gen_FeedForwardLayer = gen_factory(
        "weights",
        "bias",
        activation_fn=_getattr_if("activation_fn", _not_none),
    )
    gen_Gate = gen_factory(
        "weights",
        "bias",
        "recurrent_weights",
        peephole_weights=_getattr_if("peephole_weights", _not_none),
        activation_fn=_getattr_if("activation_fn", _not_none),
    )
    gen_Cell = gen_factory(
        "weights",
        "bias",
        "recurrent_weights",
        activation_fn=_getattr_if("activation_fn", lambda v: v is not tanh),
    )

    # gen function for a numpy ndarray.
    # Serializes the array data in human-readable decimal notation.
    def gen_ndarray_ascii(arr):
        return f"np.array({arr.tolist()}, dtype=np.{arr.dtype})"

    # gen function for a numpy ndarray.
    # Serializes the array data in a compressed binary format using base64 encoding.
    def gen_ndarray_b64bytes(arr):
        maybe_reshape = ""
        if len(arr.shape) != 1:
            maybe_reshape = f".reshape({repr(arr.shape)})"
        return (
            f"np.frombuffer(base64.b64decode({repr(base64.b64encode(arr.tobytes()))}), dtype=np.{arr.dtype})"
            + maybe_reshape
        )

    # gen function for a numpy ndarray.
    # Chooses whether to use the human-readable decimal notation
    # or the binary base64 encoding
    # based on the number of array elements (small arrays use decimal notation)
    # and the `compress_arrays` argument to `code_gen()`.
    def gen_ndarray(arr):
        if arr.shape == () or np.prod(arr.shape) < 10 or not compress_arrays:
            return gen_ndarray_ascii(arr)
        else:
            return gen_ndarray_b64bytes(arr)

    # Generic gen function that dispatches based on object type.
    # Used recursively from inside other gen_*() functions when there is nested data.
    def gen(obj):
        def default(x):
            if not allow_missing:
                raise ValueError(f"Don't know how to serialize {x.__class__.__name__}")
            return f"MISSING({x.__class__.__name__})"

        # If you write new gen_*() functions, add them and their associated type here:
        return {
            str: repr,
            int: repr,
            float: repr,
            list: lambda obj: f"[{gen_seq(obj)}]",
            tuple: lambda obj: f"({gen_seq(obj)})",
            function: lambda obj: obj.__name__,
            np.ndarray: gen_ndarray,
            NeuralNetworkEnsemble: gen_NeuralNetworkEnsemble,
            ParallelProcessor: gen_ParallelProcessor,
            NeuralNetwork: gen_NeuralNetwork,
            LSTMLayer: gen_LSTMLayer,
            FeedForwardLayer: gen_FeedForwardLayer,
            Gate: gen_Gate,
            Cell: gen_Cell,
        }.get(type(obj), default)(obj)

    return gen(obj)


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Un-pickler for MADMOM models",
        epilog="Example: madmom-unpickler.py 'NeuralNetworkEnsemble.load(madmom.models.BEATS_LSTM, ensemble_fn=average_predictions)'",
    )
    parser.add_argument(
        "object", help="The madmom object to unpickle (this expression will be eval'd)"
    )
    parser.add_argument(
        "--no-format",
        dest="format",
        action="store_false",
        default=True,
        help="Don't run the 'black' formatter on the output",
    )
    parser.add_argument(
        "--no-compress-arrays",
        dest="compress_arrays",
        action="store_false",
        default=True,
        help="Don't compress array data using base64 encoding",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        default=False,
        help="Continue code generation even if an unrecognized object is encountered (result will not run)",
    )
    args = parser.parse_args()

    # Load the object we are being asked to serialize
    obj = eval(args.object)

    if args.format and not args.compress_arrays:
        print(
            "WARNING! Uncompressed arrays + formatting can be very slow",
            file=sys.stderr,
        )

    # Perform code generation
    code = code_gen(
        obj, compress_arrays=args.compress_arrays, allow_missing=args.allow_missing
    )

    # Optionally, format the result
    if args.format:
        from black import format_str, FileMode

        code = format_str(code, mode=FileMode())

    print(code)


if __name__ == "__main__":
    main()
