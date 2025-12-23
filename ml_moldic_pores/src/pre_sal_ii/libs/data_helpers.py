import datetime as dt
import numpy as np
import pandas as pd

def pir_alt1(df, column, prefix=None, prefix_sep="_", drop=False, join=False):
    # ref.: https://stackoverflow.com/a/45313942/195417
    prefix = column if prefix is None else prefix
    dummies = (
            pd.get_dummies(
                    pd.DataFrame(df[column].tolist(), df.index).stack(),
                    prefix=prefix,
                    prefix_sep=prefix_sep,
                )
                .astype(int)
                .groupby(level=0)
                .sum()
        )

    def zero_cols(df, cols):
        df[cols] = df[cols].fillna(0).astype(int)
        return df
        
    dropper = lambda df: df.drop(column, axis=1) if drop else df[column]
    joinner = lambda df: zero_cols(df.join(dummies), dummies.columns) if join else dummies
    return joinner(dropper(df))

def pir_fast(df, column, prefix=None, prefix_sep="_", drop=False, join=False):
    # ref.: https://stackoverflow.com/a/45313942/195417
    prefix = column if prefix is None else prefix
    v = df[column].values
    l = [len(x) for x in v.tolist()]
    f, u = pd.factorize(np.concatenate(v))
    u = np.array([f"{prefix}{prefix_sep}{x}" for x in u])
    n, m = len(v), u.size
    i = np.arange(n).repeat(l)

    dummies = pd.DataFrame(
        np.bincount(i * m + f, minlength=n * m).reshape(n, m),
        df.index, u
    )

    dropper = lambda df: df.drop(column, axis=1) if drop else df[column]
    joinner = lambda df: df.join(dummies) if join else dummies
    return joinner(dropper(df))

def get_many_dummies(df, column, prefix=None, prefix_sep="_", replace=False, perf_split=500):
    exec = pir_fast if df.shape[0] < perf_split else pir_alt1
    result = exec(df, column, prefix=prefix, prefix_sep=prefix_sep, drop=replace, join=replace)
    return result

def test_get_many_dummies():
    data = np.array([
        [ "x", "y", "z" ],
        [ ["a"], ["b", "c"], [] ],
    ], dtype=object).T
    ixs = [1,2,3]
    cols = ["n", "v"]
    df = pd.DataFrame(data, ixs, cols)
    print(df)

    dms = get_many_dummies(df, "v", replace=True)
    print(dms)

    dms = get_many_dummies(df, "v", replace=True, perf_split=0)
    print(dms)

def matrix_unshear(matrix):
    matrix = np.array(matrix).transpose()
    diagonals = []
    diag_idx = []
    for line in matrix:
        first = next((i for i, x in enumerate(line) if ~np.isnan(x)), -1)
        last = next((i for i, x in enumerate(reversed(line)) if ~np.isnan(x)), -1)
        if first == -1 and last == -1:
            continue
        elif first >= 0 and last == 0:
            diagonals.append(line[first:])
            diag_idx.append(-first)
        elif first == 0 and last >= 0:
            diagonals.append(line[:-last])
            diag_idx.append(last)
        else:
            raise Exception("Invalid matrix for matrix_unshear. matrix is not sheared.")
    from scipy.sparse import diags
    result = diags(diagonals, diag_idx)
    return result

def test_matrix_unshear():
    mat = matrix_unshear([
        [np.nan, 0.6, 0.4],
        [   0.6, 0.0, 0.4],
        [   0.6, 0.0, 0.4],
        [   0.6, 0.4, np.nan],
    ])
    res = np.array([
        [1,0,0,0],
        [0,0,1,0],
        [1,0,0,0],
        [0,0,0,1],
        [0,1,0,0],
    ]) * mat
    res


def cache_dataframe_result(fn):
    import os
    import pandas as pd
    from functools import wraps
    filename = f"cache/{fn.__name__}.csv"

    @wraps(fn)
    def _fn(*args, read_cache=True, write_cache=True, converters=None, **kwargs):
        os.makedirs("./cache/", exist_ok=True)
        if read_cache and os.path.exists(filename):
            hit = True
            df = pd.read_csv(filename, index_col=0)
            if converters is not None:
                df = apply_converters(df, converters)
        else:
            hit = False
            df = fn(*args, **kwargs)
            if write_cache:
                df.to_csv(filename)
        return df, hit
    return _fn

def test_cache_dataframe_result():
    try:
        def test_fn_ret_df():
            return test_apply_converters()
        df, hit = cache_dataframe_result(test_fn_ret_df)()
        print("Cache hit:", hit)
        print(df)
        import os
        print("File present: ", os.path.isfile("test_fn_ret_df.csv"))
        df2, hit = cache_dataframe_result(test_fn_ret_df)()
        print("Cache hit:", hit)
        print(df2)
    finally:
        os.remove("test_fn_ret_df.csv")

def apply_converters(df, converters={}):
    import re
    for rx in converters:
        for col in df.columns:
            match = re.search(rx, col)
            matched = match is not None
            if matched:
                def convert(x):
                    result = converters[rx](x)
                    return result
                df[col] = df[col].apply(convert)
            else:
                pass
    return df

def test_apply_converters():
    import json
    import pandas as pd
    import ast
    df = pd.DataFrame([
        ["2022-12-25", "['marry','christmas']"],
        ["2022-12-31", "['happy','new','year']"],
    ], columns=["event_datetime", "message_array"])
    print("INPUT", df)
    df2 = apply_converters(df, {
        r"_array$": ast.literal_eval,
        r"_datetime$": pd.to_datetime,
    })
    df2["event_datetime"] = df2["event_datetime"].dt.year
    df2["message_array"] = df2["message_array"].apply(lambda x: x[0])
    print("OUTPUT", df2)
    return df2

def regex_split_column(df, column, pattern, replace=True):
    import pandas as pd
    df_result = df[column].str.extract(pattern, expand=True)
    if replace:
        df = df.drop(column, axis=1)
        df_result = pd.concat([df, df_result], axis=1)
    return df_result

def test_regex_split_column():
    df = pd.DataFrame([
        ["X", "2022-12-25 Miguel Angelo"],
        ["Y", "2022-12-31 Santos Bicudo"],
    ], columns=["letter", "string"], index=[12, 15])
    regex_split_column(
            df,
            "string",
            r"(?P<date>\d+-\d+-\d+) (?P<name>.*)",
            replace=True,
        )

def save_csv(df: pd.DataFrame, filename, force_replace=False, **kwargs):
    import os
    import hashlib
    import shutil
    import json
    os.makedirs("data", exist_ok=True)
    if force_replace or not os.path.isfile(f"data/{filename}"):
        df.to_csv(f"data/{filename}", **kwargs)
    with open(f"data/{filename}", "rb") as f:
        file_hash = hashlib.sha256()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    hash = file_hash.hexdigest()
    file_name, file_extension = os.path.splitext(filename)
    shutil.copy(
            f"data/{filename}",
            f"data/{file_name}.{hash}{file_extension}",
        )
    print(file_hash.hexdigest())
    with open(f"data/{file_name}.{hash}.txt", "w") as f:
        f.write(json.dumps({
                "shape": [*df.shape],
                "columns": [*df.columns]
            }, indent=2))

def test_save_csv():
    save_csv(None, "1.1-get-cve-data.csv")

def round_values(value, multiple):
    return (value%multiple > 0)*multiple + (value//multiple)*multiple

def lower_resolution(a: np.array, factor=28):
    """Lowers the resolution of a one dimensional array, making it blocky.
    
    For example, the array [0, 2, 4, 6, 8, 10] after applying a resolution factor 2
    becomes [1, 1, 5, 5, 9, 9].

    Args:
        a (np.array): _description_
        factor (int, optional): _description_. Defaults to 28.

    Returns:
        _type_: _description_
    """
    r = np.zeros(round_values(len(a), factor)).astype("float")
    r[0:len(a)] = a
    for k in range(1, factor):
        r[0:len(a):factor] += r[k:len(a)+k:factor]
    r = r/factor
    for k in range(1, factor):
        r[k:len(a)+k:factor] = r[0:len(a):factor]
    return r


def reject_outliers(sr):
    q1, q3 = np.quantile(sr, [0.25, 0.75])
    iqr = q3 - q1
    fl = q1 - 1.5*iqr
    fu = q3 + 1.5*iqr
    return sr[(fl <= sr) & (sr <= fu)]

def is_outliers(sr):
    q1, q3 = np.quantile(sr, [0.25, 0.75])
    iqr = q3 - q1
    fl = q1 - 1.5*iqr
    fu = q3 + 1.5*iqr
    return ~((fl <= sr) & (sr <= fu))



def convert_to_timestamp_if_needed(date):
    if isinstance(date, (dt.date, dt.datetime)):
        new_timestamp =  pd.Timestamp(date)
        new_timestamp = new_timestamp.tz_localize(None)
        return new_timestamp
    return date

def convert_to_int64_date_if_needed(date):
    if isinstance(date, (dt.date, dt.datetime)):
        new_timestamp = np.datetime64(date).astype("<M8[us]").astype("int64")
        return new_timestamp
    return date

def date_in_range_args(date, start, end):
    need_int64_date = False
    need_timestamp = False
    if isinstance(date, pd.DatetimeIndex):
        need_timestamp = True
    elif isinstance(date, pd.Timestamp):
        need_timestamp = True
    elif isinstance(date, (np.ndarray, pd.Index)):
        date = pd.to_datetime(date)
        need_timestamp = True
    elif isinstance(date, cp.ndarray):
        if date.shape == ():
            date = date.item()
        need_int64_date = True
    # elif isinstance(date, cpn.DatetimeIndex):
    #     need_int64_date = True

    if need_timestamp:
        start = convert_to_timestamp_if_needed(start)
        end = convert_to_timestamp_if_needed(end)
    if need_int64_date:
        start = convert_to_int64_date_if_needed(start)
        end = convert_to_int64_date_if_needed(end)
    return (date, start, end)

def date_in_range(date, start, end):
    date, start, end = date_in_range_args(date, start, end)
    try:
        result = (date >= start) & (date < end)
        return result
    except Exception as ex:
        print(f"type(date) = {type(date)}")
        print(f"type(start) = {type(start)}")
        print(f"type(end) = {type(end)}")
        raise ex from None

def year_filter(year):
    return lambda d: date_in_range(d, dt.date(year, 1, 1), dt.date(year+1, 1, 1))

def year2_filter(year_start, year_end):
    return lambda d: date_in_range(d, dt.date(year_start, 1, 1), dt.date(year_end+1, 1, 1))

def date2_filter(date_start, date_end):
    if isinstance(date_start, [list, tuple]):
        date_start = dt.date(*date_start)
    if isinstance(date_end, [list, tuple]):
        date_end = dt.date(*date_end)
    return lambda d: date_in_range(d, date_start, date_end)

def date_interval_filter(date_start, time_span):
    if isinstance(date_start, (list, tuple)):
        date_start = dt.date(*date_start)
    date_end = date_start + time_span
    return lambda d: date_in_range(d, date_start, date_end)

def month_filter(year, month):
    if month < 12:
        return lambda d: date_in_range(d, dt.date(year, month, 1), dt.date(year, month+1, 1))
    return lambda d: date_in_range(d, dt.date(year, 12, 1), dt.date(year+1, 1, 1))

def days_filter(year, month, day, interval_days):
    lb = dt.date(year, month, day)
    ub = lb + dt.timedelta(days=interval_days)
    return lambda d: date_in_range(d, lb, ub)


def create_year_month_intervals(start_year, start_month, end_year, end_month):
    if start_year == end_year:
        return [(start_year, m) for m in range(start_month, end_month + 1)]
    result = []
    result.extend((start_year, m) for m in range(start_month, 13))
    result.extend((y, m) for y in range(start_year + 1, end_year) for m in range(1, 13))
    result.extend((end_year, m) for m in range(1, end_month+1))
    return result

if False:
    display([
        create_year_month_intervals(2000, 2, 2000, 10),
        create_year_month_intervals(2000, 2, 2001, 10),
    ])

def bins_from_date_index(series):
    min_index = min(series.index)
    max_index = max(series.index)
    return [min_index + dt.timedelta(days=x)
            for x in range((max_index - min_index).days + 2)]

transform_ones = lambda v: v*0 + 1

import math
# import cupy as cp
# def datetime_range_cp(start_date: dt.datetime, end_date: dt.datetime, interval: dt.timedelta):
#     nums = (end_date - start_date)/interval
#     nums = math.ceil(nums)
#     items = [(start_date + interval*x) for x in range(nums)]
#     result = cp.array(np.array(items).astype("datetime64").astype("int64"))
#     return result

def datetime_range(start_date: dt.datetime, end_date: dt.datetime, interval: dt.timedelta):
    nums = (end_date - start_date)/interval
    nums = math.ceil(nums)
    items = [(start_date + interval*x) for x in range(nums)]
    result = items
    return result

if __name__ == "__main__":
    test_save_csv()
