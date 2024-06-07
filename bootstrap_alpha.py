"""
Bootstrapping sampling distributions of Krippendorff's Alpha

A fast python implementation based on `fast-krippendorff` github repo and
Krippendorff's method.

https://github.com/pln-fing-udelar/fast-krippendorff
http://afhayes.com/public/alphaboot.pdf
"""
import numpy as np

from typing import Any, Callable, Iterable, Optional, Sequence, Union

from krippendorff.krippendorff import (
    _distance_metric,
    _distances,
    _random_coincidences,
    _reliability_data_to_value_counts,
)

from krippendorff import alpha


def prepare_value_counts_and_domain(
    reliability_data: Optional[Iterable[Any]] = None,
    value_counts: Optional[np.ndarray] = None,
    value_domain: Optional[Sequence[Any]] = None,
    level_of_measurement: Union[str, Callable[..., Any]] = 'interval',
) -> tuple[np.ndarray, Sequence[Any]]:
    """
    Perform sanity check and transformation for two input arguments of `bootstrap`

    This part of code is a direct derivation from the preprocessing steps in
    `alpha` function. It takes four input arguments, `reliability_data`,
    `value_counts`, and `value_domain`, and returns only the latter two in
    their revised forms.

    See the definition of `alpha` for detailed descriptions about these
    parameters.
    https://github.com/pln-fing-udelar/fast-krippendorff/blob/main/krippendorff/krippendorff.py

    Parameters
    ----------
    reliability_data : array_like, with shape (M, N)

    value_counts : array_like, with shape (N, V)

    value_domain : array_like, with shape (V,)

    level_of_measurement : string or callable

    Returns
    -------
    value_counts : array_like, with shape (N, V)

    value_domain : array_like, with shape (V,)
    """
    if (reliability_data is None) == (value_counts is None):
        raise ValueError("Either reliability_data or value_counts must be provided, but not both.")

    # Don't know if it's a list or numpy array. If it's the latter, the truth value is ambiguous. So, ask for None.
    if value_counts is None:
        reliability_data = np.asarray(reliability_data)

        kind = reliability_data.dtype.kind
        if kind in {"i", "u", "f"}:
            # np.isnan only operates on signed integers, unsigned integers, and floats, not strings.
            found_value_domain = np.unique(reliability_data[~np.isnan(reliability_data)])
        elif kind in {"U", "S"}:  # Unicode or byte string.
            # np.asarray will coerce np.nan values to "nan".
            found_value_domain = np.unique(reliability_data[reliability_data != "nan"])
        else:
            raise ValueError(f"Don't know how to construct value domain for dtype kind {kind}.")

        if value_domain is None:
            # Check if Unicode or byte string
            if kind in {"U", "S"} and level_of_measurement != "nominal":
                raise ValueError("When using strings, an ordered value_domain is required "
                                 "for level_of_measurement other than 'nominal'.")
            value_domain = found_value_domain
        else:
            value_domain = np.asarray(value_domain)
            # Note: We do not need to test for np.nan in the input data.
            # np.nan indicates the absence of a domain value and is always allowed.
            assert np.isin(found_value_domain, value_domain).all(), \
                "The reliability data contains out-of-domain values."

        value_counts = _reliability_data_to_value_counts(reliability_data, value_domain)
    else:
        value_counts = np.asarray(value_counts)

        if value_domain is None:
            value_domain = np.arange(value_counts.shape[1])
        else:
            value_domain = np.asarray(value_domain)
            assert value_counts.shape[1] == len(value_domain), \
                "The value domain should be equal to the number of columns of value_counts."

    assert len(value_domain) > 1, "There has to be more than one value in the domain."

    return value_counts, value_domain


def bootstrap(
    reliability_data: Optional[Iterable[Any]] = None,
    value_counts: Optional[np.ndarray] = None,
    value_domain: Optional[Sequence[Any]] = None,
    level_of_measurement: Union[str, Callable[..., Any]] = 'interval',
    dtype: Any = np.float64,
    confidence_level: float = 0.95,
    num_iterations: int = 1000,
    return_bootstrap_estimates: bool = False,
    sampling_method: str = 'krippendorff',
):
    """
    Bootstrap a distribution of Krippendorff's alpha

    This algorithm is based on Krippendorff (2006), with the alpha updated
    following the coincidence matrix based implementation as in the
    `fast-krippendorff` github repo.

    https://github.com/pln-fing-udelar/fast-krippendorff

    Parameters
    ----------
    reliability_data : array_like, with shape (M, N)
        Reliability data matrix which has the rate the i coder gave to the j
        unit, where M is the number of raters and N is the unit count.  Missing
        rates are represented with `np.nan`.  If it's provided then
        `value_counts` must not be provided.

    value_counts : array_like, with shape (N, V)
        Number of coders that assigned a certain value to a determined unit,
        where N is the number of units and V is the value count.  If it's
        provided then `reliability_data` must not be provided.

    value_domain : array_like, with shape (V,)
        Possible values the units can take.  If the level of measurement is not
        nominal, it must be ordered.  If `reliability_data` is provided, then
        the default value is the ordered list of unique rates that appear.
        Else, the default value is `list(range(V))`.

    level_of_measurement : string or callable, default: `'interval'`
        Steven's level of measurement of the variable.  It must be one of
        "nominal", "ordinal", "interval", "ratio", or a callable.

    dtype: data type, default: `np.float64`
        Result and computation data-type.

    confidence_level: float, default: `0.95`,
        Designated confidence level.

    num_iterations: int, default: `1000`
        Number of boostrapping iterations.

    return_bootstrap_estimates: boolean , default: `False`
        When set true, the function will in addition return the bootstrap
        estimates (as the second component in the tuple).

    sampling_method: str, default: `'krippendorff'`
        Pair sampling methods, can be either `'krippendorff'` (Krippendorff
        2006) or `'random'` (pure random sampling).

    Returns
    -------
    confidence_interval : np.ndarray
        Induced confidence interval as an numpy array of shape (2,), in the
        form of (lower bound, upper bound).

    estimations: np.ndarray
        Bootstrap estimates.

    References
    ----------
    Hayes, Andrew F. & Krippendorff, Klaus (2007). Answering the call for a
    standard reliability measure for coding data. Communication Methods and
    Measures, 1, 77–89.

    Krippendorff, Klaus (2006). Bootstrapping Distributions for Krippendorff’s
    Alpha. http://afhayes.com/public/alphaboot.pdf
    """
    assert sampling_method in ('krippendorff', 'random')

    # Preprocess input arguments
    value_counts, value_domain = prepare_value_counts_and_domain(
        reliability_data,
        value_counts,
        value_domain,
        level_of_measurement,
    )

    distance_metric = _distance_metric(level_of_measurement)

    # Compute the unnormalized coincidence matrix and save it for later
    V = value_counts.shape[1]
    pairable = np.maximum(value_counts.sum(axis=1), 2)
    diagonals = value_counts[:, np.newaxis, :] * np.eye(V)[np.newaxis, ...]
    unnormalized_coincidences = value_counts[..., np.newaxis] * value_counts[:, np.newaxis, :] - diagonals

    # Compute `o`, `e`, and `d` following the original `fast-krippendorff` implementation
    o = np.divide(unnormalized_coincidences, (pairable - 1).reshape((-1, 1, 1)), dtype=dtype).sum(axis=0)
    n_v = o.sum(axis=0)
    e = _random_coincidences(n_v, dtype=dtype)
    d = _distances(value_domain, distance_metric, n_v, dtype=dtype)

    # Calculate the resampling probabilities based on the unnormalized coincidences
    o_raw = unnormalized_coincidences.sum(axis=0)
    prob = (o_raw / o_raw.sum()).reshape(-1)
    S = np.arange(prob.size)
    E = np.eye(prob.size)

    # Perform the actual bootstrapping process
    est = []
    for i in range(num_iterations):

        # Loop through all units
        #
        # For every unit, calculate the number of needed pairs and resample
        # judgments accordingly
        units = []
        n_pairs = (pairable * (pairable - 1) / 2).astype(int)
        for n in n_pairs:

            # Sampling approach 1: Krippendorff's method
            if sampling_method == 'krippendorff':
                idx = np.array([], dtype=np.int64)
                while len(idx) < n:
                    draw = np.random.choice(S, size=(n, 2), replace=True, p=prob)
                    idx = np.concatenate([idx, draw[draw[:, 0] != draw[:, 1], 0]])
                idx = idx[:n]

            # Sampling approach 2: random sample
            else:
                idx = np.random.choice(S, size=n, replace=True, p=prob)

            assert len(idx) == n

            sample = E[idx].sum(axis=0).reshape(o_raw.shape)
            units.append(sample)

        # From the sample, calculate the alpha value
        sampled_coincidences = 2 * np.array(units)
        o_new = np.divide(sampled_coincidences, (pairable - 1).reshape((-1, 1, 1)), dtype=dtype).sum(axis=0)
        a_new = 1 - (o_new * d).sum() / (e * d).sum()
        est.append(a_new)

    # Compute sample percentiles
    est = np.array(est)

    a = 1 - confidence_level
    res = np.percentile(est, [100 * a/2, 100 * (1 - a/2)])

    # Include bootstrap estimates as necessary
    if return_bootstrap_estimates:
        res = (res, est)

    return res


EXAMPLE_DATASET = '''
1       1       1       2       .       2
2       1       1       0       1       .
3       2       3       3       3       .
4       .       0       0       .       0
5       0       0       0       .       0
6       0       0       0       .       0
7       1       0       2       .       1
8       1       .       2       0       .
9       2       2       2       .       2
10      2       1       1       1       .
11      .       1       0       0       .
12      0       0       0       0       .
13      1       2       2       2       .
14      3       3       2       2       3
15      1       1       1       .       1
16      1       1       1       .       1
17      2       1       2       .       2
18      1       2       3       3       .
19      1       1       0       1       .
20      0       0       0       .       0
21      0       0       1       1       .
22      0       0       .       0       0
23      2       3       3       3       .
24      0       0       0       0       .
25      1       2       .       2       2
26      0       1       1       1       .
27      0       0       0       1       0
28      1       2       1       2       .
29      1       1       2       2       .
30      1       1       2       .       2
31      1       1       0       .       0
32      2       1       2       1       .
33      2       2       .       2       2
34      3       2       2       2       .
35      2       2       2       .       2
36      2       2       3       .       2
37      2       2       2       .       2
38      2       2       .       1       2
39      2       2       2       2       .
40      1       1       1       .       1
'''


def get_example_data():
    """
    Get the example reliability data as given in Hayes & Krippendorff (2007,
    Table 1)

    Returns
    -------
    dataset : np.ndarray, with shape (N, M)
        Rating matrix of N units and M raters.
    """
    rows = []
    for line in EXAMPLE_DATASET.splitlines():
        if len(line) == 0:
            continue
        row = [int(v) if v.isnumeric() else None for v in line.split()]
        rows.append(row[1:])
    return np.array(rows, dtype=float)


def demo():
    """
    Demo function
    """
    X = get_example_data()
    print('Example Data\n\n', X, '\n')

    # Input reliability data is assumed to be of the shape (raters, units)
    alpha_value = alpha(
        reliability_data=X.T,
        level_of_measurement='ordinal',
    )
    print(f"Krippendorff's alpha: {alpha_value:.6f}")

    ci, est = bootstrap(
        reliability_data=X.T,
        level_of_measurement='ordinal',
        num_iterations=10000,
        return_bootstrap_estimates=True,
        sampling_method='krippendorff',
    )
    mu = np.percentile(est, 50)
    print(f"Bootstrap estimate (Krippendorff's method): {mu:.6f} (95%-CI: [{ci[0]:.6f}, {ci[1]:.6f}])")

    ci, est = bootstrap(
        reliability_data=X.T,
        level_of_measurement='ordinal',
        num_iterations=10000,
        return_bootstrap_estimates=True,
        sampling_method='random',
    )
    mu = np.percentile(est, 50)
    print(f"Bootstrap estimate (random sample): {mu:.6f} (95%-CI: [{ci[0]:.6f}, {ci[1]:.6f}])")


if __name__ == '__main__':
    demo()