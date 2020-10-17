# BDA 696
# Karenina Zaballa

# PLEASE NOTE: This is not finished

# Import packages

import sys

import numpy as np
import pandas as pd
import statsmodels.api
from plotly import express as px
from sklearn.ensemble import RandomForestClassifier

# REFERENCES:
# Logistic Regression -
# https://towardsdatascience.com/building-a-logistic-regression-in-
# python-step-by-step-becd4d56c9c8
# Linear Regression -
# https://towardsdatascience.com/the-complete-guide-to-linear-
# regression-in-python-3d3f8f06bf8


def boolean_processing(predictor, response):
    # fill in with logistic regression
    return print("boolean processing")


def continuous_processing(predict_df, response):
    print("Continuous response pathway activated")
    # linear_regression_model_fitted

    for idx, column in enumerate(predict_df.T):
        if (np.dtype(predict_df[idx]) == str) or (np.dtype(predict_df[idx]) == bool):
            print("This variable is categorical")
        elif (np.dtype(predict_df[idx]) == int) or (np.dtype(predict_df[idx]) == float):
            print("This variable is continuous")
        else:
            print("This variable is ", np.dtype(predict_df[idx]))
        feature_name = predict_df[idx]
        predictor = statsmodels.api.add_constant(column)

        linear_regression_model = statsmodels.api.OLS(response, predictor)
        linear_regression_model_fitted = linear_regression_model.fit()
        print(f"Variable: {feature_name}")
        print(linear_regression_model_fitted.summary())

        # Get the stats
        t_value = round(linear_regression_model_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

        # Plot the figure
        fig = px.scatter(x=column, y=response, trendline="ols")
        fig.update_layout(
            title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {feature_name}",
            yaxis_title="y",
        )
    fig.show()
    fig.write_html(file=f"../../plots/lecture_6_var_{idx}.html", include_plotlyjs="cdn")

    return


def main():
    # read in Dataset here
    # df = pd.read_csv("D:\SDSU BDA\BDA 696\HW4\SD_listings_July2020.csv")
    df = pd.read_csv(" ")
    df.columns.values  # becomes array
    df_list = df.columns.values.tolist()  # becomes list
    # Check if your list is correct
    print(df_list)
    Y = df["reviews_per_month"]  # assign your response variable here
    Y.head
    X = df.loc[:, df.columns != "reviews_per_month"]
    X.head

    print("*" * 87)
    print("Testing if your response variable is continuous or boolean")
    if (np.dtype(Y) == float) or (np.dtype(Y) == int) is True:
        if np.array_equal(Y, Y.astype(bool)) is True:
            print("Your response variable is boolean.")
            boolean_processing(X, Y)
        else:
            print("Your response variable is continuous.")
            continuous_processing(X, Y)
    elif np.array_equal(Y, Y.astype(bool)) is True:
        print("Your response variable is boolean.")
        boolean_processing(X, Y)
    else:
        print(
            "Your response variable is an unidentified data type. "
            "Maybe categorical or its own object?"
        )
    print("*" * 87)
    # run weighted portion here
    # run ranking algorithm here
    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators=df.columns.size)
    # Train the model on training data
    rf.fit(df)

    print("Mean Absolute Error:", round(np.mean, 2), "degrees.")


if __name__ == "__main__":
    sys.exit(main())
