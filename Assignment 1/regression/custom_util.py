import pandas as pd     
import matplotlib.pyplot as plt #
import seaborn as sns
from typing import  List, Union, Tuple
Series = pd.core.frame.Series 
DataFrame = pd.core.frame.DataFrame # define type alias
def outlierDetection(df : DataFrame, var : str, sort : bool = True) -> DataFrame:
    """
    Utilty function for detecting basic outliers. Prints out no. of outliers, shows first 5 outliers and displays a box plot. Outliers are defined in this case by Tukey's fences, where outliers are data points above or below the upper or lower quantiles by 1.5 * IQR.

    Parameters:
        df = DataFrame to detect outliers in
        var = name of column where outlier detection is performed
        sort = should the returned dataset be sorted?
    Returns:
        Dataset which only includes outliers
    """
    Q3 = df[var].quantile(0.75)
    Q1 = df[var].quantile(0.25)
    IQR = Q3 - Q1
    UpperFence = Q3 + 1.5 * IQR
    LowerFence = Q1 - 1.5 * IQR
    mask = (df[var] > UpperFence) | (df[var] < LowerFence)
    outlier_df = df[mask]
    if len(outlier_df) == 0:
        print("No Outliers")
    else:
        print("Outliers in Series (First 5)")
        display(outlier_df.head())
        print("No. of Outliers:", len(outlier_df))
    sns.boxplot(y = var,data = df, orient= "h")
    sns.despine(left = True)
    plt.title(f"Box Plot of {var}")
    plt.show()
    if sort:
        return outlier_df.sort_values(var)
    else:
        return outlier_df

def removeOutliers(data: Union[DataFrame, Series], cols : Union[List[str], None] = None) -> DataFrame:
    df_type = type(data)
    assert df_type is DataFrame or df_type is Series, "data should either be a pandas dataframe or series"
    assert cols is None or type(cols) is list, "Either provide None, or a list of col names"
    if cols is not None:
        data = data[cols]
    Q3 = data.quantile(0.75)
    Q1 = data.quantile(0.25)
    IQR = Q3 - Q1
    print("Shape Before Removing Outliers:", data.shape)
    data = data[~((data < (Q1 - (1.5 * IQR))) | (data > (Q3 + (1.5 * IQR)))).any(axis=1)]
    print("Shape After Removing Outliers:", data.shape)
    return data