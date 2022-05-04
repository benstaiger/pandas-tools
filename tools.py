import pandas as pd


def equal_subset(subset, superset, key) -> bool:
    subset = subset.set_index(key)
    superset = superset.set_index(key)

    # Check that all the rows in subset exist in the superset
    missing_in_new = subset.index.difference(superset.index)
    if len(missing_in_new) > 0:
        print("Data missing from new table:")
        print(missing_in_new)
        print()
        return False

    superset = superset.loc[subset.index]
    if not superset.equals(subset):
        print("Values differ for some rows:")
        print(subset.compare(superset))
        print()
        return False
    return True


def copy_data_from_key(data_from, data_to, key):
    data_from = data_from.set_index(key)
    data_to = data_to.set_index(key)
    to_copy = data_from.index
    assert len(data_from.index.difference(data_to.index)) == 0  # copy all
    data_to.loc[to_copy] = data_from.loc[to_copy]
    data_to.reset_index(inplace=True)
    return data_to


def test_equal_subset():
    data = pd.DataFrame(
        {
            "key1": [1, 1, 2, 2, 3],
            "key2": [1, 2, 1, 2, 0],
            "val1": [1, 2, 2, 3, 4],
            "val2": [1, 2, 2, 3, 4],
            "val3": [1, 2, 2, 3, 4],
        }
    )
    assert equal_subset(data, data, ["key1", "key2"])

    more_data = pd.DataFrame(
        {
            "key1": [5, 3],
            "key2": [1, 2],
            "val1": [2, 2],
            "val2": [3, 6],
            "val3": [4, 2],
        }
    )
    superset = pd.concat([data, more_data], ignore_index=True)
    assert equal_subset(data, superset, ["key1", "key2"])

    superset = pd.concat([data, more_data], ignore_index=True)
    assert not equal_subset(superset, data, ["key1", "key2"])

    values_differ = superset.copy()
    values_differ.loc[0, "val2"] = 3
    values_differ.loc[2, "val3"] = 6
    assert not equal_subset(data, values_differ, ["key1", "key2"])


def test_copy():
    data = pd.DataFrame(
        {
            "key1": [1, 1, 2, 2, 3],
            "key2": [1, 2, 1, 2, 0],
            "val1": [1, 2, 2, 3, 4],
            "val2": [1, 2, 2, 3, 4],
            "val3": [1, 2, 2, 3, 4],
        }
    )
    more_data = pd.DataFrame(
        {
            "key1": [5, 3],
            "key2": [1, 2],
            "val1": [2, 2],
            "val2": [3, 6],
            "val3": [4, 2],
        }
    )
    superset = pd.concat([data, more_data], ignore_index=True)
    values_differ = data.copy()
    values_differ.loc[0, "val2"] = 3
    values_differ.loc[2, "val3"] = 6

    assert not equal_subset(values_differ, superset, ["key1", "key2"])
    new_superset = copy_data_from_key(values_differ, superset, ["key1", "key2"])
    assert equal_subset(values_differ, new_superset, ["key1", "key2"])


if __name__ == "__main__":
    test_equal_subset()
    test_copy()
