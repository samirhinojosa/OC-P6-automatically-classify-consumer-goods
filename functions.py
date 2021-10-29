import io
import gc
import pandas as pd
from math import prod


def df_analysis(df, name_df,
                *args, **kwargs):
    """
    Method used to analyze on the DataFrame.

    Parameters:
    -----------------
        df (pandas.DataFrame): Dataset to analyze
        name_df (str): Dataset name

        *args, **kwargs:
        -----------------
            columns (list): Dataframe keys in list format
            flag (str): Flag to show complete information about
                        the dataset to analyse "complete" shows
                        all information about the dataset

    Returns:
    -----------------
        None.
        Print the analysis on the Dataset.
    """

    # Getting the variables
    columns = kwargs.get("columns", None)

    # Getting the variables
    # tha analysis type are ["complete", "header"]
    # By defaut "complete"
    analysis_type = kwargs.get("analysis_type", None)

    ordered_columns = [
        "name", "type", "records", "unique",
        "# NaN", "% NaN", "mean", "min",
        "25%", "50%", "75%", "max", "std"
    ]

    # Calculating the memory usage based on dataframe.info()
    buf = io.StringIO()
    df.info(buf=buf)
    memory_usage = buf.getvalue().split("\n")[-2]

    if df.empty:
        print("The", name_df, "dataset is empty. Please verify the file.")
    else:
        empty_cols = [col for col in df.columns
                      if df[col].isna().all()]  # identifying empty columns
        df_rows_duplicates = df[df.duplicated()]  # identifying duplicates rows

        # Creating a dataset based on Type object and records by columns
        type_cols = df.dtypes.apply(lambda x: x.name).to_dict()
        df_resume = pd.DataFrame(list(type_cols.items()),
                                 columns=["name", "type"])
        df_resume["records"] = list(df.count())
        df_resume["unique"] = list(df.nunique())
        df_resume["# NaN"] = list(df.isnull().sum())
        df_resume["% NaN"] = list(((df.isnull().sum()
                                    / len(df.index))*100).round(2))

        # Printing the analysis header
        print("\nAnalysis Header of", name_df, "dataset")
        print(70*"-")

        print("- Dataset shape:\t\t\t", df.shape[0], "rows and",
              df.shape[1], "columns")
        print("- Total of NaN values:\t\t\t", df.isna().sum().sum())
        print("- Percentage of NaN:\t\t\t",
              round((df.isna().sum().sum()/prod(df.shape))
                    * 100, 2), "%")
        print("- Total of full duplicates rows:\t",
              df_rows_duplicates.shape[0])

        if df.dropna(axis="rows", how="all").shape[0] < df.shape[0]:
            print("- Total of empty rows:\t\t\t",
                  df.shape[0] - df.dropna(axis="rows", how="all"))
        else:
            print("- Total of empty rows:\t\t\t", "0")

        print("- Total of empty columns:\t\t", len(empty_cols))

        if len(empty_cols) == 1 or len(empty_cols) >= 1:
            print("\t+ The empty column is:\t\t", empty_cols)
        else:
            None

        print("- Unique indexes:\t\t\t", df.index.is_unique)
        print("- Memory usage:\t\t\t\t",
              memory_usage.split("memory usage: ")[1])

        if columns is not None:
            if df.size == df.drop_duplicates(columns).size:
                print("\n- The key(s):", columns, "is not present\
                      multiple times in the dataframe.\n\
                      It CAN be used as a primary key.")
            else:
                print("\n- The key(s):", columns, "is present multiple\
                      times in the dataframe.\n  It CANNOT be used as a\
                      primary key.")

        # Enabling the visualizacion of
        # all columns, rows, cell and floar format
        print("\n")
        pd.set_option("display.max_rows",
                      None)  # show full of showing rows
        pd.set_option("display.max_columns",
                      None)  # show full of showing cols
        pd.set_option("display.max_colwidth",
                      None)  # show full width of showing cols
        pd.set_option("display.float_format",
                      lambda x: "%.5f" % x)  # show full float in cell

        if analysis_type == "complete" or analysis_type is None:
            pass
        elif analysis_type == "header":
            pass

        pd.reset_option("display.max_rows")  # reset max of showing rows
        pd.reset_option("display.max_columns")  # reset max of showing cols
        pd.reset_option("display.max_colwidth")  # reset width of showing cols
        pd.reset_option("display.float_format")  # reset float format in cell
