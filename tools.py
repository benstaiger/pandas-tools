import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.datasets import load_iris


def plot_stacked_hist(data1, data2, *, stacked=False, prop=True, xlab=""):
    plt.title(f"{data1.name} vs. {data2.name}")
    _, bins = np.histogram(pd.concat([data1, data2], ignore_index=True), bins=10)
    if stacked:
        plt.hist(
            [data1, data2],
            edgecolor="black",
            color=["blue", "red"],
            histtype="barstacked",
            alpha=0.5,
            bins=20,
            density=prop,
            label=[f"{data1.name}", f"{data2.name}"],
        )
    else:
        def plot_data(data, color, with_normal=True):
            plt.hist(
                data,
                edgecolor="black",
                color=color,
                alpha=0.5,
                bins=bins,
                density=prop,
                label=f"{data.name}",
            )
            if with_normal:
                continuous = np.linspace(bins[0], bins[-1], num=len(bins)*5)
                sigma = data.std()
                mu = data.mean()
                plt.plot(
                    continuous,
                    1 / (sigma * np.sqrt(2 * np.pi))
                    * np.exp(-((continuous - mu) ** 2) / (2 * sigma ** 2)),
                    linewidth=2,
                    color=color,
                )
        plot_data(data1, "blue")
        plot_data(data2, "red")
    plt.legend(loc="upper right")
    plt.xlabel(xlab)
    if prop:
        plt.ylabel("Distribution of Occurences")
    else:
        plt.ylabel("# of Occurences")
    plt.tight_layout()
    plt.show()


def chi_square(counts):
    row_total = counts.sum(axis=1)
    col_total = counts.sum(axis=0)
    props = col_total / col_total.sum()
    expected = pd.DataFrame(
        [[p * r for p in props.values] for r in row_total.values],
        index=row_total.index,
        columns=props.index,
    )
    dif = counts - expected
    stats = dif * dif / expected
    return stats.sum().sum(), expected, stats


def propagate_val(data, key_col, val_col):
    data.set_index(key_col, inplace=True)
    keys = data.index.unique()
    for k in keys:
        vals = data.loc[k][val_col].unique()
        vals = [v for v in vals if v != ""]
        if len(vals) > 1:
            print(f"{k} has different values {vals}")
        if len(vals) == 1:
            data.loc[k][val_col] = vals[0]
    data.reset_index()
    return data


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


def test_hist():
    iris = load_iris(as_frame=True)
    iris_data = iris["data"]
    t0 = iris_data["sepal length (cm)"][iris["target"] == 0]
    t1 = iris_data["sepal length (cm)"][iris["target"] == 1]
    t0.name = iris["target_names"][0]
    t1.name = iris["target_names"][1]
    plot_stacked_hist(t0, t1, xlab="iris sepal length (cm)")


if __name__ == "__main__":
    test_equal_subset()
    test_copy()
    # test_hist()
