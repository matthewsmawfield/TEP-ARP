"""JSON output sanitiser for pipeline step outputs.

Python's ``json.dump`` serialises ``float('nan')`` and infinities as the
bare tokens ``NaN``/``Infinity``, which are not valid JSON and are
rejected by strict parsers (jq, JavaScript ``JSON.parse``, Go, Rust).
Pipeline outputs must remain spec-compliant so downstream consumers can
read them without a lenient parser, so step writers pass their summary
objects through ``json_safe`` first.
"""

import math


def json_safe(obj):
    """Return ``obj`` with NaN and non-finite floats replaced by None.

    Recurses into dicts, lists and tuples; every other type is returned
    unchanged.  ``float('nan')``, ``float('inf')`` and
    ``float('-inf')`` become ``None`` rather than a sentinel number so
    that "no measurement" is distinguishable from a measured value.
    NumPy scalars and arrays are converted through ``.item()`` /
    ``.tolist()`` so that values such as ``np.int64`` or ``np.bool_``
    serialise instead of raising ``TypeError``; the numpy check is
    duck-typed on the module name so this module needs no numpy import.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if type(obj).__module__.startswith("numpy"):
        scalar = getattr(obj, "item", None)
        if callable(scalar):
            try:
                return json_safe(scalar())
            except (ValueError, TypeError):
                pass
        lst = getattr(obj, "tolist", None)
        if callable(lst):
            return json_safe(lst())
    return obj
