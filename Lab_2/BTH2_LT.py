import pandas as pd

#================PANDAS===============
print("===============PANDAS===============")
# 1/ Pandas Series is essentially a one-dimensional array, equipped with an index which labels its entries.
# We can create a Series object, for example, by converting a list (called diameters)
# [4879,12104,12756,6792,142984,120536,51118,49528]
print("==========Cau 1==========")
diameters = [4879,12104,12756,6792,142984,120536,51118,49528]
ds = pd.Series(diameters)
print(ds)
print(ds[0])

# 2/ By default entries of a Series are indexed by consecutive integers, but we can specify a more meaningful index.
# The numbers in the above Series give diameters (in kilometers) of planets of the Solar System, so it is sensible
# to use names of the planet as index values:
#Index=[“Mercury”, “Venus”, “Earth”, “Mars”, “Jupyter”, “Saturn”, “Uranus”, “Neptune”]
print("==========Cau 2==========")
diameters = [4879,12104,12756,6792,142984,120536,51118,49528]
index = ["Mercury", "Venus", "Earth", "Mars", "Jupyter", "Saturn", "Uranus", "Neptune"]
ds = pd.Series(diameters, index=index)
print(ds)

#3/ Find diameter of Earth?
print("==========Cau 3==========")
diameters = [4879,12104,12756,6792,142984,120536,51118,49528]
index = ["Mercury", "Venus", "Earth", "Mars", "Jupyter", "Saturn", "Uranus", "Neptune"]
earth_diameter = ds["Earth"]
print("Diameter of Earth:", earth_diameter)

#4/ Find diameters from “Mercury” to “Mars” basing on data on 2/
print("==========Cau 4==========")
diameters_mercury_to_mars = ds["Mercury":"Mars"]
print(diameters_mercury_to_mars)

#5/  Find diameters of “Earth”, “Jupyter” and “Neptune” (with one command)?
print("==========Cau 5==========")
diameters_selected = ds[["Earth", "Jupyter", "Neptune"]]
print(diameters_selected)

#6/ I want to modify the data in diameters. Specifically, I want to add the diameter of Pluto 2370.
# Saved the new data in the old name “diameters”.
print("==========Cau 6==========")
planet_diameters = pd.Series(['2370'], index=['Pluto'])
print(planet_diameters)

#7/ Pandas DataFrame is a two-dimensional array equipped with one index labeling its rows, and another
# labeling its columns.There are several ways of creating a DataFrame. One of them is to use a dictionary of lists.
# Each list gives values of a column
# of the DataFrame, and dictionary keys give column labels:
#“diameter”=[4879,12104,12756,6792,142984,120536,51118,49528,2370]
#“avg_temp”=[167,464,15,-65,-110, -140, -195, -200, -225]
#“gravity”=[3.7, 8.9, 9.8, 3.7, 23.1, 9.0, 8.7, 11.0, 0.7]
# Create a pandas DataFrame, called planets.
print("==========Cau 7==========")
data = {
    "diameter": [4879, 12104, 12756, 6792, 142984, 120536, 51118, 49528, 2370],
    "avg_temp": [167, 464, 15, -65, -110, -140, -195, -200, -225],
    "gravity": [3.7, 8.9, 9.8, 3.7, 23.1, 9.0, 8.7, 11.0, 0.7]
}

planets = pd.DataFrame(data)
print(planets)
planets.index = ["Mercury", "Venus", "Earth", "Mars", "Jupyter", "Saturn", "Uranus", "Neptune", "Pluto"]
print(planets)

#8/ Get the first 3 rows of “planets”.
print("==========Cau 8==========")
first_three_rows = planets.head(3)
print(first_three_rows)

#9/ Get the last 2 rows of “planets”.
print("==========Cau 9==========")
last_two_row = planets.tail(2)
print(last_two_row)

#10/ Find the name of columns of “planets”
print("==========Cau 10==========")
print(planets.columns)

#11/ Since we have not specified an index for rows, by default it consists of consecutive integers.
# We can change it by modifying
# the index by using the name of the corresponding planet. Check the index after modifying.
print("==========Cau 11==========")
data = {
    "diameter": [4879, 12104, 12756, 6792, 142984, 120536, 51118, 49528, 2370],
    "avg_temp": [167, 464, 15, -65, -110, -140, -195, -200, -225],
    "gravity": [3.7, 8.9, 9.8, 3.7, 23.1, 9.0, 8.7, 11.0, 0.7]
}
planets = pd.DataFrame(data)
planets.index = ["Mercury", "Venus", "Earth", "Mars", "Jupyter", "Saturn", "Uranus", "Neptune", "Pluto"]
print(planets.index)

#12/ How to get the gravity of all planets in “planets”?
print("==========Cau 12==========")
print(planets["gravity"])

#13/ How to get the gravity and diameter of all planets in “planets”?
print("==========Cau 13==========")
print(planets[['gravity', 'diameter']])

#14/ Find the gravity of Earth using loc?
print("==========Cau 14==========")
print(planets.loc["Earth", "gravity"])

#15/ Similarly, find the diameter and gravity of Earth?
print("==========Cau 15==========")
print(planets.loc["Earth" , ["diameter", "gravity"]])

#16/ Find the gravity and diameter from Earth to Saturn?
print("==========Cau 16==========")
print(planets.loc["Earth":"Saturn", ["gravity", "diameter"]])

#17/ Check (using Boolean) all the planets in “planets” that have diameter >1000?
print("==========Cau 17==========")
print(planets[planets['diameter'] > 1000])

#18/ Select all planets in “planets” that have diameter>100000?
print("==========Cau 18==========")
print(planets[planets['diameter'] > 100000])

#19/ Select all planets in “planets” that satisfying avg-temp>0 and gravity>5.
print("==========Cau 19==========")
print(planets[(planets['avg_temp'] > 0) & (planets['gravity'] > 5)])

#20/ Sort values of diameter in “diameters” in ascending order.
print("==========Cau 20==========")
print(sorted(diameters))

#21/ Sort values of diameter in “diameters” in descending order.
print("==========Cau 21==========")
print(sorted(diameters, reverse=True))

#22/ Sort using the “gravity” column in descending order in “planets”.
print("==========Cau 22==========")
print(planets.sort_values(by='gravity', ascending=False))

#23/ Sort values in the “Mercury” row.
print("==========Cau 23==========")
mercury_row = planets.loc["Mercury"]
sorted_mercury_row = mercury_row.sort_values()
print(sorted_mercury_row)


import matplotlib.pyplot as plt
import seaborn as sns

#===============SEABORNS================
print("===============SEABORNS===============")
#1/ Seaborn is Python library for visualizing data. Seaborn uses matplotlib to create graphics, but it provides
# tools that make it much easier to create several types of plots. In particular, it is simple to use seaborn
# with pandas dataframes.
print("==========Cau 1==========")
tips = sns.load_dataset("tips")
sns.set_style("whitegrid")
g = sns.lmplot(x="tip",
               y="total_bill",
               data=tips,
               aspect=2)
g = (g.set_axis_labels("Tip","Total bill(USD)").set(xlim=(0,10),ylim=(0,100)))
plt.title("title")
plt.show()

#2/ Display name of datasets.
print("==========Cau 2==========")
dataset_names = sns.get_dataset_names()
print(dataset_names)

#3/ How can get a pandas dataframe with the data.
print("==========Cau 3==========")
iris_df = sns.load_dataset("iris")
print(iris_df.head())

#4/ How to produce a scatter plot showing the bill amount on the axis and the tip amount on the axis?
print("==========Cau 4==========")
tips = sns.load_dataset("tips")
sns.scatterplot(data=tips, x="total_bill", y="tip")
plt.title("Scatter Plot of Total Bill vs Tip Amount")
plt.xlabel("Total Bill ($)")
plt.ylabel("Tip Amount ($)")
plt.show()

#5/ By default, seaborn uses the original matplotlib settings for fonts, colors etc.How to modify
# font=1.2 and color=darkgrid?
print("==========Cau 5==========")
sns.set_context("notebook", font_scale=1.2)
sns.set_style("darkgrid")
tips = sns.load_dataset("tips")
sns.scatterplot(data=tips, x="total_bill", y="tip")
plt.title("Scatter Plot of Total Bill vs Tip Amount")
plt.xlabel("Total Bill ($)")
plt.ylabel("Tip Amount ($)")
plt.show()

#6/ We can use the values in the “day” column to assign marker colors. How?
print("==========Cau 6==========")
sns.set_context("notebook", font_scale=1.2)
sns.set_style("darkgrid")
tips = sns.load_dataset("tips")
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day", palette="deep")
plt.title("Scatter Plot of Total Bill vs Tip Amount by Day")
plt.xlabel("Total Bill ($)")
plt.ylabel("Tip Amount ($)")
plt.legend(title='Day')
plt.show()

#7/ Next, we set different marker sizes based on values in the “size” column.
print("==========Cau 7==========")
sns.set_context("notebook", font_scale=1.2)
sns.set_style("darkgrid")
tips = sns.load_dataset("tips")
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day", size="size", sizes=(20, 200), palette="deep")
plt.title("Scatter Plot of Total Bill vs Tip Amount by Day and Size")
plt.xlabel("Total Bill ($)")
plt.ylabel("Tip Amount ($)")
plt.legend(title='Day')
plt.show()

#8/ We can also split the plot into subplots based on values of some column. Below we create two subplots,
# each displaying data for a different value of the “time” column
print("==========Cau 8==========")
sns.set_context("notebook", font_scale=1.2)
sns.set_style("darkgrid")
tips = sns.load_dataset("tips")
g = sns.FacetGrid(tips, col="time", height=5, aspect=1)
g.map_dataframe(sns.scatterplot, x="total_bill", y="tip", hue="day", size="size", sizes=(20, 200), palette="deep")
g.set_axis_labels("Total Bill ($)", "Tip Amount ($)")
g.set_titles(col_template="{col_name} Time")
plt.show()

#9/ We can subdivide the plot even further using values of the “sex” column
print("==========Cau 9==========")
sns.set_context("notebook", font_scale=1.2)
sns.set_style("darkgrid")
tips = sns.load_dataset("tips")
g = sns.FacetGrid(tips, row="sex", col="time", height=5, aspect=1)
g.map_dataframe(sns.scatterplot, x="total_bill", y="tip", hue="day", size="size", sizes=(20, 200), palette="deep")
g.set_axis_labels("Total Bill ($)", "Tip Amount ($)")
g.set_titles(row_template="{row_name}", col_template="{col_name} Time")  # Set titles for each subplot
plt.show()