# BDA 696 - Karenina Zaballa
# Week 1 - HW1
# installing libraries as needed
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

# Read in data files
iris_df = pd.read_csv("D:/SDSU BDA/BDA 696/HW1/HW1-Data/iris.data.csv")
# iris_df1 = pd.read_csv("D:/SDSU BDA/BDA 696/HW1/HW1-Data/iris.names.csv")
iris_df.head

# Descriptive Statistics
iris_df.describe()
# RESULTS: - has no column headings
#             5.1         3.5         1.4         0.2
# count  149.000000  149.000000  149.000000  149.000000
# mean     5.848322    3.051007    3.774497    1.205369
# std      0.828594    0.433499    1.759651    0.761292
# min      4.300000    2.000000    1.000000    0.100000
# 25%      5.100000    2.800000    1.600000    0.300000
# 50%      5.800000    3.000000    4.400000    1.300000
# 75%      6.400000    3.300000    5.100000    1.800000
# max      7.900000    4.400000    6.900000    2.500000


# Plots:
#   1)Scatter plot
# 		add column headings to have something to anchor on when graphing
iris_df.columns = [
    "Sepal Length",
    "Sepal Width",
    "Petal Length",
    "Petal Width",
    "Class",
]
iris_df.head
# 		plot here
fig = px.scatter(
    iris_df, x="Sepal Length", y="Sepal Width", color="Class", title="Iris Test Plot"
)
fig.show()

#   2) Histogram
# 		group by class and have a different color for each class
fig0 = px.histogram(iris_df, x="Class", color="Class")
fig0.show()

#   3) Line graph
# 		group by class and have a different color for each class
#       This looks messy because it was a line graph that had to go
#       through each point in the scatterplot. I will have to refine this
#       later to just reflect a trendline.
fig1 = px.line(iris_df, x="Petal Length", y="Petal Width", color="Class")
fig1.show()

results = px.get_trendline_results(fig1)
print(results)

#   4) Pie chart
#
fig3 = px.pie(
    iris_df,
    names="Class",
    values="Petal Width",
    color="Class",
    title="Petal Width by class in different colors",
)
fig3.show()

#   5) histogram, scatter and rug
#   double graphs
fig4 = px.scatter(
    iris_df,
    x="Sepal Length",
    y="Sepal Width",
    marginal_x="histogram",
    marginal_y="rug",
    color="Class",
)
fig4.show()

#   6) Different scatter-line plot styles
#
iris_df.head
fig5 = go.Figure()
fig5.add_trace(
    go.Scatter(
        x=iris_df["Sepal Length"],
        y=iris_df["Sepal Width"],
        mode="lines+markers",
        name="lines+markers",
    )
)
fig5.add_trace(
    go.Scatter(
        x=iris_df["Sepal Length"],
        y=iris_df["Petal Length"],
        mode="lines+markers",
        name="lines+markers",
    )
)
fig5.add_trace(
    go.Scatter(
        x=iris_df["Sepal Length"],
        y=iris_df["Petal Width"],
        mode="markers",
        name="markers",
    )
)
fig5.show()

#   7) Violin, box, with main scatter plot
fig6 = px.scatter(
    iris_df,
    x=iris_df["Sepal Length"],
    y=iris_df["Sepal Width"],
    color="Class",
    marginal_x="box",
    marginal_y="violin",
    title="Click on the legend items!",
)
fig6.show()

#   Normalize and transform
# Normalizer does not accept qualitative data
X = iris_df[["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]]
X_trans = Normalizer().transform(X)
# set norm to mean of sepal length and transform
# it by that
#
# Is my logic correct here? I am thinking in
# order to truly normalize something, each max or mean
# or whatever function has to be applied to each column
# transformer = Normalizer(norm=max(X["Sepal Length"]),
# max(X["Sepal Width"]),
# max(X["Petal Length"]),
# max(X["Petal Width"])).transform(X) <-- this caused an error
# transformer = Normalizer(norm=X["Sepal Length"].
# mean()).transform(X)<-- this caused an error
# transformer
# X_trans = transformer.transform(X)
# X_trans
# tried to adjust this to # of rows and columns
# It works but I don't fully understand its effects
X_trans, y = make_classification(
    n_samples=149,
    n_features=4,
    n_informative=2,
    n_redundant=0,
    random_state=0,
    shuffle=False,
)

# This one seems to be easier to explain because it has less
# parameters since we have 149 rows and 4 columns
X_trans, y = make_classification(n_samples=149, n_features=4)
# apply classifier
clf = RandomForestClassifier(max_depth=4, random_state=0)
clf.fit(X_trans, y)
print(clf.predict([[1, 0, 0, 1]]))
# RESULT: [1]


X_train, X_test, y_train, y_test = train_test_split(X_trans, y, random_state=0)

# Here is where I had changed it to Normalizer since I saw the bottom to here
# The pipeline can be used as any other estimator and avoids
# leaking the test set into the train set.
pipe = Pipeline([("scaler", Normalizer()), ("svc", SVC())])
pipe.fit(X_train, y_train)

pipe.score(X_test, y_test)
# RESULT: 0.9210526315789473
