# BDA 696
# Karenina Zaballa


# Import packages

import sys

import numpy as np
import pandas as pd
import statsmodels.api
from plotly import express as px
from plotly import figure_factory as ff
from pygments.lexers import go
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# REFERENCES:
# Logistic Regression -
# https://towardsdatascience.com/building-a-logistic-regression-in-
# python-step-by-step-becd4d56c9c8
# Linear Regression -
# https://towardsdatascience.com/the-complete-guide-to-linear-
# regression-in-python-3d3f8f06bf8


def main():
    # read in Dataset here
    df = pd.read_csv(" ")
    # df = pd.read_csv(" ")
    df = df.dropna(axis=1, how="any")
    df.columns.values  # becomes array
    df_list = df.columns.values.tolist()  # becomes list
    # Check if your list is correct
    print(df_list)
    Y = df["response_variable"]  # assign your response variable here
    Y.head
    X = df.loc[:, df.columns != "_responsevariable"]
    X.head

    print("*" * 87)
    print("Testing if your response variable is continuous or boolean")

    if isinstance(Y, float) or isinstance(Y, int):
        if Y.nunique() > 2:
            print("Your response variable is continuous.")
            response_cont = True
            continuous_processing(X, Y, response_cont)
        else:
            print("Your response variable is boolean.")
            response_cont = False
            boolean_processing(X, Y, response_cont)
    elif isinstance(Y, bool):
        print("Your response variable is boolean.")
        response_cont = False
        boolean_processing(X, Y, response_cont)
    else:
        print("Your response variable datatype is ", np.dtype(Y))
    print("*" * 87)
    # run weighted portion here
    # run ranking algorithm here
    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators=df.columns.size)
    # Train the model on training data
    rf.fit(x=X, y=Y)

    print("Mean Absolute Error:", round(np.mean, 2), "degrees.")


def boolean_processing(predict_df, response, response_bool):
    for column in predict_df:
        if isinstance(column, str) or isinstance(column, bool):
            predictor_cont = False
            print("This variable is categorical")
        else:
            if column.nunique() > 2:
                predictor_cont = True
                print("This variable is continuous")
            else:
                predictor_cont = False
                print("This variable is categorical")

        logreg = LogisticRegression()
        rfe = RFE(logreg, predict_df.columns.size)
        rfe = rfe.fit(predict_df, response.values.ravel())
        print(rfe.support_)
        print(rfe.ranking_)

        if predictor_cont is False:

            conf_matrix = confusion_matrix(predict_df, response)

            fig_no_relationship = go.Figure(
                data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
            )
            fig_no_relationship.update_layout(
                title="Categorical Predictor by Categorical "
                "Response (without relationship)",
                xaxis_title="Response",
                yaxis_title="Predictor",
            )
            fig_no_relationship.show()
            fig_no_relationship.write_html(
                file="../../../plots/HW4_predictor_heat_map_no_relation.html",
                include_plotlyjs="cdn",
            )

            fig_no_relationship = go.Figure(
                data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
            )
            fig_no_relationship.update_layout(
                title="Categorical Predictor by Categorical "
                "Response (with relationship)",
                xaxis_title="Response",
                yaxis_title="Predictor",
            )
            fig_no_relationship.show()
            fig_no_relationship.write_html(
                file="../../../plots/HW4_predictor_heat_map_yes_relation.html",
                include_plotlyjs="cdn",
            )
        elif predictor_cont is True:
            # Group data together
            hist_data = [predictor_cont[column], response[column]]
            group_labels = ["Response = 0", "Response = 1"]

            # Create distribution plot with custom bin_size
            fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
            fig_1.update_layout(
                title="Continuous Predictor by Categorical Response",
                xaxis_title="Predictor",
                yaxis_title="Distribution",
            )
            fig_1.show()
            fig_1.write_html(
                file="../../../plots/HW4_cat_response_cont_predictor_dist_plot.html",
                include_plotlyjs="cdn",
            )

            fig_2 = go.Figure()
            for curr_hist, curr_group in zip(hist_data, group_labels):
                fig_2.add_trace(
                    go.Violin(
                        x=np.repeat(curr_group),
                        y=curr_hist,
                        name=curr_group,
                        box_visible=True,
                        meanline_visible=True,
                    )
                )
            fig_2.update_layout(
                title="Continuous Predictor by Categorical Response",
                xaxis_title="Response",
                yaxis_title="Predictor",
            )
            fig_2.show()
            fig_2.write_html(
                file="../../../plots/HW4_cat_response_cont_predictor_violin_plot.html",
                include_plotlyjs="cdn",
            )


def continuous_processing(predict_df, response, response_bool):
    print("Continuous response pathway activated")
    # linear_regression_model_fitted

    predict_df
    for column in predict_df:
        if isinstance(column, str) or isinstance(column, bool):
            print("This variable is categorical")
        elif isinstance(column, int) or isinstance(column, float):
            if column.nunique() > 2:
                print("This variable is continuous")
            else:
                print("This variable is categorical")
        else:
            print("This variable is ", np.dtype(predict_df[column]))

        feature_name = predict_df[column]
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
        fig.write_html(file=f"../../plots/HW4_{column}.html", include_plotlyjs="cdn")

        fig = px.violin(x=column, y=response, trendline="ols")
        fig.update_layout(
            title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {feature_name}",
            yaxis_title="y",
        )
        fig.show()
        fig.write_html(file=f"../../plots/HW4_{column}.html", include_plotlyjs="cdn")

    # should put another plot here

    return


if __name__ == "__main__":
    sys.exit(main())
