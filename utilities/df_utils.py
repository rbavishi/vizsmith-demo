from typing import Dict

import pandas as pd


def check_if_id_like(df, cardinalities, attribute):
    """
    Adopted from the Lux Project:
    https://github.com/lux-org/lux/blob/952d3c59389c83eb20eff80e9ff93cdd58a542af/lux/utils/utils.py#L74
    :param df:
    :param cardinalities:
    :param attribute:
    :return:
    """
    import re

    # Strong signals
    # so that aggregated reset_index fields don't get misclassified
    high_cardinality = cardinalities[attribute] > 500
    attribute_contain_id = re.search(r"id|ID|iD|Id", str(attribute)) is not None
    almost_all_vals_unique = cardinalities[attribute] >= 0.98 * len(df)
    is_string = pd.api.types.is_string_dtype(df[attribute])
    if is_string:
        #  For string IDs, usually serial numbers or codes with alphanumerics have a consistent length (eg.,
        #  CG-39405) with little deviation. For a high cardinality string field but not ID field (like Name or Brand),
        #  there is less uniformity across the string lengths.
        if len(df) > 50:
            sampled = df[attribute].sample(50, random_state=99)
        else:
            sampled = df[attribute]
        str_length_uniformity = sampled.apply(lambda x: type(x) == str and len(x)).std() < 3
        return (
                high_cardinality
                and (attribute_contain_id or almost_all_vals_unique)
                and str_length_uniformity
        )
    else:
        if len(df) >= 2:
            series = df[attribute]
            diff = series.diff()
            evenly_spaced = all(diff.iloc[1:] == diff.iloc[1])
        else:
            evenly_spaced = True
        if attribute_contain_id:
            almost_all_vals_unique = cardinalities[attribute] >= 0.75 * len(df)
        return high_cardinality and (almost_all_vals_unique or evenly_spaced)


def compute_df_metadata(df) -> Dict:
    """
    Adopted from the Lux Project:
    https://github.com/lux-org/lux/blob/2525d7115b9e6fa2de537dcbd088ab7c8990f904/lux/executor/PandasExecutor.py#L404
    and
    https://github.com/lux-org/lux/blob/2525d7115b9e6fa2de537dcbd088ab7c8990f904/lux/executor/PandasExecutor.py#L514
    :param df:
    :return:
    """
    from pandas.api.types import is_datetime64_any_dtype as is_datetime

    # precompute statistics
    unique_values = {}
    cardinality = {}
    data_types = {}
    raw_data_types = {}
    has_null = {}

    for attribute in df.columns:
        if isinstance(attribute, pd._libs.tslibs.timestamps.Timestamp):
            # If timestamp, make the dictionary keys the _repr_ (e.g., TimeStamp('2020-04-05 00.000')--> '2020-04-05')
            attribute_repr = str(attribute._date_repr)
        else:
            attribute_repr = attribute

        if df.dtypes[attribute] != "float64" or df[attribute].isnull().values.any():
            try:
                unique_values[attribute_repr] = list(df[attribute].unique())
            except TypeError:
                unique_values[attribute_repr] = list(df[attribute])

            cardinality[attribute_repr] = len(unique_values[attribute])
        else:
            cardinality[attribute_repr] = 999  # special value for non-numeric attribute

    if not pd.api.types.is_integer_dtype(df.index):
        index_column_name = '__index'
        unique_values[index_column_name] = list(df.index)
        cardinality[index_column_name] = len(df.index)

    for attr in list(df.columns):
        if is_datetime(df[attr]):
            data_types[attr] = "temporal"
        elif isinstance(attr, pd._libs.tslibs.timestamps.Timestamp):
            data_types[attr] = "temporal"
        elif pd.api.types.is_float_dtype(df.dtypes[attr]):
            # int columns gets coerced into floats if contain NaN
            convertible2int = pd.api.types.is_integer_dtype(df[attr].convert_dtypes())
            if (
                    convertible2int
                    and cardinality[attr] != len(df)
                    and (len(df[attr].convert_dtypes().unique()) < 20)
            ):
                if len(df) < 20:
                    data_types[attr] = "categorical/quantitative"
                else:
                    data_types[attr] = "categorical"
            else:
                data_types[attr] = "quantitative"

        elif pd.api.types.is_integer_dtype(df.dtypes[attr]):
            # See if integer value is quantitative or nominal by checking if the ratio of cardinality/data size is
            # less than 0.4 and if there are less than 10 unique values
            if cardinality[attr] / len(df) < 0.25:
                if cardinality[attr] <= 5:
                    data_types[attr] = "categorical"
                else:
                    data_types[attr] = "categorical/quantitative"
            elif cardinality[attr] < 20:
                if check_if_id_like(df, cardinality, attr):
                    data_types[attr] = "categorical/quantitative/id"
                else:
                    data_types[attr] = "categorical/quantitative"
            elif check_if_id_like(df, cardinality, attr):
                data_types[attr] = "id"
            else:
                data_types[attr] = "quantitative"

            if check_if_id_like(df, cardinality, attr):
                data_types[attr] = "id"
        # Eliminate this clause because a single NaN value can cause the dtype to be object
        elif pd.api.types.is_string_dtype(df.dtypes[attr]):
            if cardinality[attr] / len(df) < 0.25:
                data_types[attr] = "categorical"
            elif cardinality[attr] < 20:
                if check_if_id_like(df, cardinality, attr):
                    data_types[attr] = "categorical/nominal/id"
                else:
                    data_types[attr] = "categorical/nominal"
            else:
                data_types[attr] = "nominal"
            if check_if_id_like(df, cardinality, attr):
                data_types[attr] = "id"
        # check if attribute is any type of datetime dtype
        elif pd.api.types.is_datetime64_any_dtype(df.dtypes[attr]) or pd.api.types.is_period_dtype(df.dtypes[attr]):
            data_types[attr] = "temporal"
        else:
            data_types[attr] = "nominal"

        raw_dtypes = set()
        for val in df[attr]:
            if pd.isnull(val) is not True:
                dtype = type(val)
                if pd.api.types.is_float_dtype(dtype):
                    raw_dtypes.add(float)
                elif pd.api.types.is_integer_dtype(dtype):
                    raw_dtypes.add(int)
                elif pd.api.types.is_string_dtype(dtype):
                    raw_dtypes.add(str)
                elif pd.api.types.is_bool_dtype(dtype):
                    raw_dtypes.add(bool)
                elif pd.api.types.is_array_like(val):
                    raw_dtypes.add('array')
                else:
                    raw_dtypes.add(type(val))

        raw_data_types[attr] = frozenset(raw_dtypes)
        has_null[attr] = df[attr].isnull().values.any()

    return {
        'cardinality': cardinality,
        'unique_values': unique_values,
        'high_level_data_types': data_types,
        'low_level_data_types': raw_data_types,
        'has_null': has_null
    }
