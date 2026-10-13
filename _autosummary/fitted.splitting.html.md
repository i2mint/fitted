# fitted.splitting

Train/test split tools

### Functions

| [`keys_aligned_list`](#fitted.splitting.keys_aligned_list)(iterable_spec, keys)         | Get an iterable that is aligned with the keys iterable, and verify that it is so.   |
|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| [`train_test_split_keys`](#fitted.splitting.train_test_split_keys)(keys[, key_to_tag, ...]) | Split keys into train and test lists.                                               |

### fitted.splitting.keys_aligned_list(iterable_spec, keys)

Get an iterable that is aligned with the keys iterable, and verify that it is so.

```pycon
>>> keys_aligned_list(lambda x: x * 2, keys=[1, 2, 3])
[2, 4, 6]
>>> keys_aligned_list([2, 4, 6], keys=[1, 2, 3])
[2, 4, 6]
>>> assert keys_aligned_list(None, keys=[1, 2, 3]) is None
```

* **Parameters:**
  * **iterable_spec**
  * **keys**
* **Returns:**

### fitted.splitting.train_test_split_keys(keys, key_to_tag=None, key_to_group=None, test_size=None, train_size=None, random_state=None, n_splits=1)

Split keys into train and test lists.

The train_keys and test_keys are disjoint and taken from keys.

Specifying key_to_tag (a function or iterable) ensures that tags will be
well distributed in both train and test.

Specifying key_to_group (a function or iterable) ensures **on the contrary**
that keys of a same group will be entirely in train **or (exclusive)** in
test – not both.

* **Parameters:**
  * **keys** ([`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable)) – keys to be split
  * **key_to_tag** (`Union`[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable), [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – keys-aligned iterable of tags (a.k.a y/classes in
    sklearn speak) or function to compute these from keys
  * **key_to_group** (`Union`[[`Callable`](https://docs.python.org/3/library/typing.html#typing.Callable), [`Iterable`](https://docs.python.org/3/library/typing.html#typing.Iterable), [`None`](https://docs.python.org/3/builtins/constants.html#None)]) – keys-aligned iterable of groups or function to compute
    these from keys
    :return a train_keys, test_keys pair (all elements of keys) if n_splits=1,
    and a generator of such pairs if not.

Note that in the doctest below, we take keys=[7, 14, 21, …] to show that
it’s not about [0, 1, 2, …] indices only, but ANY keys
(even non numerical – like filepaths, DB selectors, etc.)

```pycon
>>> keys = range(7, 7 + 100 * 7, 7)  # [7, 14, 21, ..., 700]
>>> def mod5(x):
...     return x % 5
>>> train_keys, test_keys = train_test_split_keys(keys, key_to_group=mod5,
...     train_size=.5, random_state=42)
```

Observe here that though `train_size=.5`, the proportion is not 50/50.
That’s because the group constraint, imposed by the key_to_group argument
produces only 5 groups.

```pycon
>>> len(train_keys), len(test_keys)
(40, 60)
```

But especially, see that though there’s a lot of train and test indices,
within train, there’s only 2 unique groups (all 0 or 3 modulo 5)
and only 3 unique groups (1, 2, 4 modulo 5) within test indices.

```pycon
>>> assert set(map(mod5, train_keys)) == {0, 3}
>>> assert set(map(mod5, test_keys)) == {1, 2, 4}
```
