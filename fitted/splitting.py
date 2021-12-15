"""Train/test split tools"""

from typing import Iterable, Callable, Union
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from i2.signatures import call_forgivingly


def train_test_split_keys(
    keys: Iterable,
    key_to_tag: Union[Callable, Iterable, None] = None,
    key_to_group: Union[Callable, Iterable, None] = None,
    test_size=None,
    train_size=None,
    random_state=None,
    n_splits=1,
):
    """Split keys into train and test lists.

    The train_keys and test_keys are disjoint and taken from keys.

    Specifying key_to_tag (a function or iterable) ensures that tags will be
    well distributed in both train and test.

    Specifying key_to_group (a function or iterable) ensures **on the contrary**
    that keys of a same group will be entirely in train **or (exclusive)** in
    test -- not both.

    :param keys: keys to be split
    :param key_to_tag: keys-aligned iterable of tags (a.k.a y/classes in
    sklearn speak) or function to compute these from keys
    :param key_to_group: keys-aligned iterable of groups or function to compute
    these from keys
    :return a train_keys, test_keys pair (all elements of keys) if n_splits=1,
    and a generator of such pairs if not.

    Note that in the doctest below, we take keys=[7, 14, 21, ...] to show that
    it's not about [0, 1, 2, ...] indices only, but ANY keys
    (even non numerical -- like filepaths, DB selectors, etc.)

    >>> keys = range(7, 7 + 100 * 7, 7)  # [7, 14, 21, ..., 700]
    >>> def mod5(x):
    ...     return x % 5
    >>> train_keys, test_keys = train_test_split_keys(keys, key_to_group=mod5,
    ...     train_size=.5, random_state=42)

    Observe here that though `train_size=.5`, the proportion is not 50/50.
    That's because the group constraint, imposed by the key_to_group argument
    produces only 5 groups.

    >>> len(train_keys), len(test_keys)
    (40, 60)

    But especially, see that though there's a lot of train and test indices,
    within train, there's only 2 unique groups (all 0 or 3 modulo 5)
    and only 3 unique groups (1, 2, 4 modulo 5) within test indices.

    >>> assert set(map(mod5, train_keys)) == {0, 3}
    >>> assert set(map(mod5, test_keys)) == {1, 2, 4}

    """
    splitter = call_forgivingly(
        GroupShuffleSplit, **locals()
    )  # calls GroupShuffleSplit on relevant inputs

    keys = np.array(list(keys))
    y = keys_aligned_list(key_to_tag, keys)
    groups = keys_aligned_list(key_to_group, keys)
    if groups is None:
        groups = range(len(keys))

    n = splitter.get_n_splits(keys, y, groups)
    if n == 1:
        train_idx, test_idx = next(splitter.split(keys, y, groups))
        return keys[train_idx], keys[test_idx]
    else:

        def gen():
            for train_idx, test_idx in splitter.split(keys, y, groups):
                yield keys[train_idx], keys[test_idx]

        return gen()


def keys_aligned_list(iterable_spec, keys):
    """Get an iterable that is aligned with the keys iterable, and verify that it is so.

    >>> keys_aligned_list(lambda x: x * 2, keys=[1, 2, 3])
    [2, 4, 6]
    >>> keys_aligned_list([2, 4, 6], keys=[1, 2, 3])
    [2, 4, 6]
    >>> assert keys_aligned_list(None, keys=[1, 2, 3]) is None

    :param iterable_spec:
    :param keys:
    :return:
    """
    if iterable_spec is None:
        return None
    elif isinstance(iterable_spec, Callable):
        return list(map(iterable_spec, keys))
    elif isinstance(iterable_spec, Iterable):
        iterable_spec = list(iterable_spec)
        assert len(iterable_spec) == len(keys)
        return iterable_spec
    else:
        raise TypeError(
            f'Unknown iterable_spec type ({type(iterable_spec)}): {iterable_spec}'
        )
