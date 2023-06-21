import logging
from typing import List, Literal, Dict

import pandas as pd


class SegmentGuard:
    """
    The main class for detecting issues in your data
    """


    def find_issues(self, data: pd.DataFrame, features: List[str], metric: str, metric_mode: Literal["min", "max"]="max", feature_types: Dict[str, Literal["raw", "nominal", "ordinal", "numerical"]]={}):
        """
        Find segments that are classified badly by your model.
        
        :param data: A pandas dataframe containing your data.
        :param features: A list of columns that contains features to feed into your model but also metadata.
        :param metric: A column that contains the per sample evaluation metric you want to use for finding problems.
        :param metric_mode: What do you optimize your metric for? max is the right choice for accuracy while e.g. min is good for regression error.
        
        """
        
        df = data[features + [metric]]
        
        # Try to infer the column dtypes
        dataset_length = len(df)

        for col in features:
            col_dtype = df[col].dtype
            
            if col_dtype == "object" and col not in feature_types:
                num_unique_values = len(df[col].unique())
                if num_unique_values / dataset_length > 0.5:
                    logging.warning(f"Feature {col} was inferred as referring to raw data. If this is not the case, please specify in feature_types!")
                    feature_types[col] = "raw"
                else:
                    logging.warning(f"Feature {col} was inferred as being categorical. Will be treated as nominal by default. If ordinal specify in feature_types!")
                    feature_types[col] = "nominal"
            elif col not in feature_types:
                logging.warning(f"Feature {col} will be treated as numerical value. You can override this by specifying feature_types.")
                feature_types[col] = "numerical"
            else:
                assert feature_types[col] in ("raw", "nominal", "ordinal", "numerical")


        



        print(feature_types)

    def report(self):
        """
        Create an interactive report on the found issues in spotlight.
        """
        pass
        # spotlight.show(issues=[])
    
