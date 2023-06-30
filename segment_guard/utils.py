def infer_feature_types(features, given_feature_types, df):
    """
    Infer the datatypes of certain features based on their column data.
    :param features: The features that have to be inferred.
    :param given_feature_types: Feature types that are already defined by the user.
    :param df: The dataframe containing all the data.
    """

    feature_types = {}
    for col in features:
        col_dtype = df[col].dtype

        if col_dtype == "object" and col not in given_feature_types:
            num_unique_values = len(df[col].unique())
            if num_unique_values / len(df) > 0.5:
                print(
                    f"Feature {col} was inferred as referring to raw data. If this is not the case, please specify in feature_types!"
                )
                feature_types[col] = "raw"
            else:
                print(
                    f"Feature {col} was inferred as being categorical. Will be treated as nominal by default. If ordinal specify in feature_types and feature_orders!"
                )
                feature_types[col] = "nominal"
        elif col not in given_feature_types:
            print(
                f"Feature {col} will be treated as numerical value. You can override this by specifying feature_types."
            )
            feature_types[col] = "numerical"
        else:
            assert given_feature_types[col] in (
                "raw",
                "nominal",
                "ordinal",
                "numerical",
            )
            feature_types[col] = given_feature_types[col]
    return feature_types
