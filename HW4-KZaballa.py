# # BDA 696
# # Karenina Zaballa
#
# # Import package
# import sys
#
# import numpy as np
# import pandas as pd
# import statsmodels.api
# from plotly import express as px
# from plotly import figure_factory as ff
# from plotly import graph_objects as go
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix
#
# # REFERENCES:
# # Logistic Regression -
# # https://towardsdatascience.com/building-a-logistic-regression-in-
# # python-step-by-step-becd4d56c9c8
# # Linear Regression -
# # https://towardsdatascience.com/the-complete-guide-to-linear-
# # regression-in-python-3d3f8f06bf8
#
#
# def continuous_processing(predict_df, response, response_bool):
#     print("Continuous response pathway activated")
#     # linear_regression_model_fitted
#
#     predict_df
#     for column in predict_df:
#         if isinstance(column, str) or isinstance(column, bool):
#             print("This variable is categorical")
#         elif isinstance(column, int) or isinstance(column, float):
#             if column.nunique() > 2:
#                 print("This variable is continuous")
#             else:
#                 print("This variable is categorical")
#         else:
#             print("This variable is ", np.dtype(predict_df[column]))
#
#         feature_name = predict_df[column]
#         predictor = statsmodels.api.add_constant(column)
#
#         linear_regression_model = statsmodels.api.OLS(response, predictor)
#         linear_regression_model_fitted = linear_regression_model.fit()
#         print(f"Variable: {feature_name}")
#         print(linear_regression_model_fitted.summary())
#
#         # Get the stats
#         t_value = round(linear_regression_model_fitted.tvalues[1], 6)
#         p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])
#
#         # Plot the figure
#         fig = px.scatter(x=column, y=response, trendline="ols")
#         fig.update_layout(
#             title=f"Variable: {feature_name}: (t-value={t_value})
#             (p-value={p_value})",
#             xaxis_title=f"Variable: {feature_name}",
#             yaxis_title="y",
#         )
#         fig.show()
#         fig.write_html(file=f"../../plots/HW4_{column}.html", include_plotlyjs="cdn")
#
#         fig = px.violin(x=column, y=response, trendline="ols")
#         fig.update_layout(
#             title=f"Variable: {feature_name}: (t-value={t_value})
#             (p-value={p_value})",
#             xaxis_title=f"Variable: {feature_name}",
#             yaxis_title="y",
#         )
#         fig.show()
#         fig.write_html(file=f"../../plots/HW4_{column}.html", include_plotlyjs="cdn")
#
#     # should put another plot here
#
#     return
#
#
# def boolean_processing(predict_df, response, response_bool):
#     for column in predict_df:
#         if isinstance(column, str) or isinstance(column, bool):
#             predictor_cont = False
#             print("This variable is categorical")
#         else:
#             if column.nunique() > 2:
#                 predictor_cont = True
#                 print("This variable is continuous")
#             else:
#                 predictor_cont = False
#                 print("This variable is categorical")
#
#         logreg = LogisticRegression()
#         rfe = RFE(logreg, predict_df.columns.size)
#         rfe = rfe.fit(predict_df, response.values.ravel())
#         print(rfe.support_)
#         print(rfe.ranking_)
#
#         if predictor_cont is False:
#
#             conf_matrix = confusion_matrix(predict_df, response)
#
#             fig_no_relationship = go.Figure(
#                 data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
#             )
#             fig_no_relationship.update_layout(
#                 title="Categorical Predictor by Categorical "
#                 "Response (without relationship)",
#                 xaxis_title="Response",
#                 yaxis_title="Predictor",
#             )
#             fig_no_relationship.show()
#             fig_no_relationship.write_html(
#                 file="../../../plots/HW4_predictor_heat_map_no_relation.html",
#                 include_plotlyjs="cdn",
#             )
#
#             fig_no_relationship = go.Figure(
#                 data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
#             )
#             fig_no_relationship.update_layout(
#                 title="Categorical Predictor by Categorical "
#                 "Response (with relationship)",
#                 xaxis_title="Response",
#                 yaxis_title="Predictor",
#             )
#             fig_no_relationship.show()
#             fig_no_relationship.write_html(
#                 file="../../../plots/HW4_predictor_heat_map_yes_relation.html",
#                 include_plotlyjs="cdn",
#             )
#         elif predictor_cont is True:
#             # Group data together
#             hist_data = [predictor_cont[column], response[column]]
#             group_labels = ["Response = 0", "Response = 1"]
#
#             # Create distribution plot with custom bin_size
#             fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
#             fig_1.update_layout(
#                 title="Continuous Predictor by Categorical Response",
#                 xaxis_title="Predictor",
#                 yaxis_title="Distribution",
#             )
#             fig_1.show()
#             fig_1.write_html(
#                 file="../../../plots/HW4_cat_response_cont_predictor_dist_plot.html",
#                 include_plotlyjs="cdn",
#             )
#
#             fig_2 = go.Figure()
#             for curr_hist, curr_group in zip(hist_data, group_labels):
#                 fig_2.add_trace(
#                     go.Violin(
#                         x=np.repeat(curr_group),
#                         y=curr_hist,
#                         name=curr_group,
#                         box_visible=True,
#                         meanline_visible=True,
#                     )
#                 )
#             fig_2.update_layout(
#                 title="Continuous Predictor by Categorical Response",
#                 xaxis_title="Response",
#                 yaxis_title="Predictor",
#             )
#             fig_2.show()
#             fig_2.write_html(
#                 file="../../../plots/HW4_cat_response_cont_predictor_violin_plot.html",
#                 include_plotlyjs="cdn",
#             )
#
#
# def data_prep(df, response_col):
#     # drop empty columns
#     df.dropna(axis=1, how="all", inplace=True)
#
#     # drop rows with NaN
#     df.dropna(inplace=True)
#
#     cat_thresh = 0.01
#     nominal_thresh = 6
#
#     # separate predictors and response
#     predictors = df.drop(response_col, axis=1)
#     response = df[response_col]
#
#     # determine the response type
#     response_type = "cont"
#     if len(response.unique()) == 2:
#         response_type = "bool"
#
#     # separate predictors into categorical and continious features
#     df_cat = df.loc[:, df.dtypes == object]
#     df_cont = df.loc[:, df.dtypes != object]
#
#     # drop metadata features
#     metadata_cols = []
#     for col in df_cat.columns:
#         if len(df_cat[col].unique()) > cat_thresh * df.shape[0]:
#             metadata_cols.append(col)
#
#     df_cat = df_cat.drop(metadata_cols, axis=1)
#
#     # extract categorical features encoded numerically
#     more_cat_features = []
#     more_cat_features_col = []
#     for col in df_cont.columns:
#         if len(df_cont[col].unique()) < nominal_thresh:
#             more_cat_features_col.append(col)
#             more_cat_features.append(df_cont[col])
#
#     df_cont = df_cont.drop(more_cat_features_col, axis=1)
#     if more_cat_features:
#         more_cat_features.insert(0, df_cat)
#     df_cat = pd.concat(more_cat_features, axis=1)
#
#     return df_cont, df_cat
#
#
# # def bin_col(df_, col, response_col, bins=10):
# #     pop_mean = df[response_col].mean()
# #     df_[col] = df_[col].fillna(0)
# #     labels = list(range(1, bins + 1))
# #     x = np.array(df_[col])
# #     bin_cnt, bin_edges = np.histogram(x, bins=bins)
# #     df_["binned"] = pd.cut(
# #         df_[col], bins=list(bin_edges), labels=labels, include_lowest=True
# #     )
# #
# #     # calculate unweighted response mean
# #     mean_diff = df_.groupby("binned").mean()[col].sort_index() - pop_mean
# #     unweighted_score = (mean_diff ** 2).sum()
# #
# #     # calculate weighted response mean
# #     bin_prop = bin_cnt / df_.shape[0]
# #     weighted_score = sum(np.multiply(np.array(mean_diff), np.array(bin_prop)) ** 2)
# #     # print(unweighted_score)
# #     # print(weighted_score)
# #     return unweighted_score, weighted_score
# #
# #
# # bin_col(df, "reviews_per_month", "number_of_reviews")
#
#
# def main():
#
#     # Read in your dataset
#     # must have continuous and categorical variables
#     df = pd.read_csv("D:\SDSU BDA\BDA 696\HW4\SD_listings_July2020.csv")
#
#     # df = pd.read_csv(" ")
#
#     # drop empty columns first because otherwise it will
#     drop all rows if we run the next line
#     df.dropna(axis=1, how="all", inplace=True)
#
#     # drop all rows that have null values
#     df = df.dropna(axis=1, how="any")
#     df.columns.values  # becomes array
#     df_list = df.columns.values.tolist()  # becomes list
#     # Check if your list is correct
#     print(df.columns)
#     Y = df["number_of_reviews"]  # assign your response variable here
#     Y.head()
#     # this should automatically assign your predictors to another dataframe
#     X = df.loc[:, df.columns != "number_of_reviews"]
#     X.head()
#
#     print("*" * 87)
#     print("Testing if your response variable is continuous or boolean")
#
#     # isinstance() did not work for me
#     # this should cancel out booleans that present themselves as ints
#     # and floats if they are not type object/strings
#     for i in df:
#         if len(Y.unique()) == 2:
#             print("Your response variable is boolean.")
#             boolean_processing(X, Y, True)
#         elif Y.dtype == np.int64 or Y.dtype == np.float:
#             print("Your response variable is continuous.")
#             continuous_processing(X, Y, True)
#         else:
#             print("Your response variable was not detected {}".format(Y.dtype))
#             print("#" * 20, Y.dtype)
#
#     #
#     print("*" * 87)
#
#     cat_columns = list(X.loc[:, X.dtypes == object].columns)
#     X_numeric = X.drop(cat_columns, axis=1)
#
#     rf = RandomForestClassifier(n_estimators=df.columns.size)
#
#     # run weighted portion here
#     # run ranking algorithm here
#     # Instantiate model with 1000 decision trees
#     rf = RandomForestClassifier(n_estimators=df.columns.size)
#     # Train the model on training data
#     rf.fit(x=X, y=Y)
#
#     print("Mean Absolute Error:", round(np.mean, 2), "degrees.")
#
#
# if __name__ == "__main__":
#     sys.exit(main())
