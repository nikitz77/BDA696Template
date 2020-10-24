# BDA 696 - Karenina Zaballa
# Midterm
# Import package


import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def data_prep(df, response_col):
    # drop empty columns
    df.dropna(axis=1, how="all", inplace=True)

    # drop rows with NaN
    df.dropna(inplace=True)

    cat_thresh = 0.01
    nominal_thresh = 6

    # separate predictors and response
    # predictors = df.drop(response_col, axis=1)
    # response = df[response_col]

    # determine the response type
    # response_type = "cont"
    # if len(response.unique()) == 2:
    #     response_type = "bool"

    # separate predictors into categorical and continuous features
    df_cat = df.loc[:, df.dtypes == object]
    df_cont = df.loc[:, df.dtypes != object]

    # drop metadata features
    metadata_cols = []
    for col in df_cat.columns:
        if len(df_cat[col].unique()) > cat_thresh * df.shape[0]:
            metadata_cols.append(col)

    df_cat = df_cat.drop(metadata_cols, axis=1)

    # extract categorical features encoded numerically
    more_cat_features = []
    more_cat_features_col = []
    for col in df_cont.columns:
        if len(df_cont[col].unique()) < nominal_thresh:
            more_cat_features_col.append(col)
            more_cat_features.append(df_cont[col])

    df_cont = df_cont.drop(more_cat_features_col, axis=1)
    if more_cat_features:
        more_cat_features.insert(0, df_cat)
    df_cat = pd.concat(more_cat_features, axis=1)
    return df_cont, df_cat


def fill_na(df):
    if isinstance(df, pd.Series):
        return df.fillna(0)
    else:
        return np.array([value if value is not None else 0 for value in df])


def cat_correlation(x, y, bias_correction=True, tschuprow=False):
    """
    Calculates correlation statistic for categorical-categorical association.
    The two measures supported are:
    1. Cramer'V ( default )
    2. Tschuprow'T

    SOURCES:
    1.) CODE: https://github.com/MavericksDS/pycorr
    2.) Used logic from:
        https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
        to ignore yates correction factor on 2x2
    3.) Haven't validated Tschuprow

    Bias correction and formula's taken from : https://www.researchgate.net/publication/
    270277061_A_bias-correction_for_Cramer's_V_and_Tschuprow's_T

    Wikipedia for Cramer's V: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    Wikipedia for Tschuprow' T: https://en.wikipedia.org/wiki/Tschuprow%27s_T
    Parameters:
    -----------
    x : list / ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    bias_correction : Boolean, default = True
    tschuprow : Boolean, default = False
               For choosing Tschuprow as measure
    Returns:
    --------
    float in the range of [0,1]
    """
    corr_coeff = np.nan
    # try:
    x, y = fill_na(x), fill_na(y)
    crosstab_matrix = pd.crosstab(x, y)
    n_observations = crosstab_matrix.sum().sum()

    yates_correct = True
    if bias_correction:
        if crosstab_matrix.shape == (2, 2):
            yates_correct = False

    chi2, _, _, _ = stats.chi2_contingency(crosstab_matrix, correction=yates_correct)
    phi2 = chi2 / n_observations

    # r and c are number of categories of x and y
    r, c = crosstab_matrix.shape
    if bias_correction:
        phi2_corrected = max(0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1))
        r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
        c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
        if tschuprow:
            corr_coeff = np.sqrt(
                phi2_corrected / np.sqrt((r_corrected - 1) * (c_corrected - 1))
            )
            return corr_coeff
        corr_coeff = np.sqrt(phi2_corrected / min((r_corrected - 1), (c_corrected - 1)))
        return corr_coeff
    if tschuprow:
        corr_coeff = np.sqrt(phi2 / np.sqrt((r - 1) * (c - 1)))
        return corr_coeff
    corr_coeff = np.sqrt(phi2 / min((r - 1), (c - 1)))
    return corr_coeff
    # except Exception as ex:
    #  print(ex)
    # if tschuprow:
    #  warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
    # else:
    #   warnings.warn("Error calculating Cramer's V", RuntimeWarning)
    # return corr_coeff


def bin_col_midterm(df_, col1, col2, response_col, bins=10):
    pop_mean = df_[response_col].mean()
    # df_[col1] = df_[col1].fillna(0)
    # df_[col2] = df_[col2].fillna(0)

    labels = list(range(1, bins + 1))
    x1 = np.array(df_[col1])
    bin_cnt1, bin_edges1 = np.histogram(x1, bins=bins)
    df_["binned1"] = pd.cut(
        df_[col1], bins=list(bin_edges1), labels=labels, include_lowest=True
    )

    labels = list(range(1, bins + 1))
    x2 = np.array(df_[col2])
    bin_cnt2, bin_edges2 = np.histogram(x2, bins=bins)
    df_["binned2"] = pd.cut(
        df_[col2], bins=list(bin_edges2), labels=labels, include_lowest=True
    )

    print(df_["binned1"].head())
    #     # calculate unweighted response mean
    mean_diff1 = df_.groupby("binned1").mean()[col1].sort_index() - pop_mean
    mean_diff2 = df_.groupby("binned2").mean()[col2].sort_index() - pop_mean
    unweighted_score = (mean_diff1 ** 2).sum() + (mean_diff2 ** 2).sum()

    #     print(mean_diff1)
    #     print(mean_diff2)

    #     # calculate weighted response mean
    bin_prop1 = bin_cnt1 / df_.shape[0]
    bin_prop2 = bin_cnt2 / df_.shape[0]
    weighed_mean_diff1 = np.multiply(np.array(mean_diff1), np.array(bin_prop1)) ** 2
    weighed_mean_diff2 = np.multiply(np.array(mean_diff2), np.array(bin_prop2)) ** 2

    weighted_score1 = np.nansum(weighed_mean_diff1)
    weighted_score2 = np.nansum(weighed_mean_diff2)
    weighted_score = weighted_score1 + weighted_score2

    print("****")
    print(weighed_mean_diff1)
    print(weighed_mean_diff2)

    print("****")
    print(unweighted_score)
    print(weighted_score)
    df_.drop(["binned1", "binned2"], axis=1, inplace=True)

    return unweighted_score, weighted_score


def cramers_v2(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def main():
    # read in Dataset here
    # must have continuous and categorical/nominal variables
    # df = pd.read_csv(r"D:\SDSU BDA\BDA 696\HW4\SD_listings_July2020.csv")
    df = pd.read_csv(" ")
    df = df.dropna(axis=1, how="any")

    Y = df["response"]  # assign your response column here
    # must automatically assign your predictors
    # X = df.loc[:, df.columns != "response"]

    # for columns in X.columns:
    #    print(columns)

    # Explore your data
    print(df.info())
    print("*" * 87)
    print("Testing if your response variable is continuous or boolean")

    response_cat = " "
    if len(Y.unique()) == 2:
        print("Your response variable is boolean.")
        response_cat = True
    elif Y.dtype == np.int64 or Y.dtype == np.float64 or Y.dtype == np.complex64:
        print("Your response variable is continuous.")
        response_cat = False
    else:
        print("Your response variable was not detected {}".format(Y.dtype))
        print("#" * 20, Y.dtype)

    print("*" * 87)

    print("prepping your data")

    df_cont, df_cat = data_prep(df, "number_of_reviews")
    # data_prep(df, Y)

    # Generate list
    print(df_cont.columns.tolist())
    print(df_cat.columns.tolist())

    nominal_thresh = 6
    # Generate correlation matrix for all continuous variables
    # Pearson's Correlation here
    # Table for continuous continuous
    if response_cat:
        if response_cat:
            print("Do Cramers here")
            for i in range(len(df_cat.columns)):
                print(i)
                square_root = cramers_v2(df_cat.iloc[i], df_cat.iloc[i + 1])
                print(square_root)
        else:
            print("You have no nominal/nominal variables")
    else:
        # print cont/cont table
        df_cont.corr()
        # build cont/cont table
        plt.figure(figsize=(12, 8))
        sns.heatmap(df_cont.corr(), annot=True)

        #
        df_cat_nominal = df_cat.loc[:, df.dtypes != object]

        df_nominal_cnt = (df_cat_nominal.nunique() < nominal_thresh).reset_index()
        nominal_1 = list(df_nominal_cnt.loc[df_nominal_cnt[0] is True]["index"])
        nominal_2 = list(df_nominal_cnt.loc[df_nominal_cnt[0] is False]["index"])

        df_cat_nominal_less_six = df_cat_nominal[nominal_1]
        df_cat_nominal_plus_six = df_cat_nominal[nominal_2]

        df_corr = pd.concat([df_cont, df_cat_nominal_less_six], axis=1).corr(
            method="kendall"
        )
        df_corr = df_corr[list(df_cat_nominal_less_six.columns)].loc[
            list(df_cont.columns)
        ]

        # table and plot continuous response and categorical predictors less than 6
        plt.figure(figsize=(12, 8))
        sns.heatmap(df_corr, annot=True)

        # table plot continuous response and categorical predictors more than 6
        df_cat_nominal_plus_six
        df_corr = pd.concat([df_cont, df_cat_nominal_plus_six], axis=1).corr()
        df_corr = df_corr[list(df_cat_nominal_plus_six.columns)].loc[
            list(df_cont.columns)
        ]
        plt.figure(figsize=(12, 8))
        sns.heatmap(df_corr, annot=True)
        #

        # weighted part
        # LINK TO HW 4 MAPS

        cont_cont_unweighted = []
        cont_cont_weighted = []
        response_col = Y
        for i in list(df_cont.columns):
            col_corr_unweighted = []
            col_corr_weighted = []
            for j in list(df_cont.columns):
                print(i, j)
                unweighted_score, weighted_score = bin_col_midterm(
                    df_cont, i, j, response_col
                )
                col_corr_unweighted.append(unweighted_score)
                col_corr_weighted.append(weighted_score)
            cont_cont_unweighted.append(col_corr_unweighted)
            cont_cont_weighted.append(col_corr_weighted)
        corr_mtx_4a = pd.DataFrame(
            cont_cont_weighted, columns=df_cont.columns, index=df_cont.columns
        )
        corr_mtx_4a

        sns.heatmap(corr_mtx_4a)

        rank_unweighted_single = []
        rank_weighted_single = []
        for j in len(df.columns):
            print(j)
            unweighted_score, weighted_score = bin_col_midterm(df, j, j + 1, Y)
            rank_unweighted_single.append(unweighted_score)
            rank_weighted_single.append(weighted_score)


if __name__ == "__main__":
    sys.exit(main())
