
import pandas as pd
from typing import List, Literal

class SegmentGuard:
    """
    The main class for detecting issues in your data
    """


    def find_issues(data: pd.Dataframe, features: List[str], metric: str, metric_mode: Literal["min", "max"]="max"):
        """
        Find segments that are classified badly by your model.
        
        :param data: A pandas dataframe containing your data.
        :param features: A list of columns that contains features to feed into your model but also metadata.
        :param metric: A column that contains the per sample evaluation metric you want to use for finding problems.
        :param metric_mode: What do you optimize your metric for? max is the right choice for accuary while e.g. min is good for regression error.
        
        """
        pass


    def report():
        """
        Create an interactive report on the found issues in spotlight.
        """
        pass
    
