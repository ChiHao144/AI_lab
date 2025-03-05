import pandas as pd

#1/ Show the first 5 lines of tsv file.
print("==========Cau 1==========")
file_path = '04_gap-merged.tsv'
data = pd.read_csv(file_path, sep='\t')
print(data.head(5))

#2/ Find the number of row and column of this file.
print("==========Cau 2==========")
num_rows, num_columns = data.shape
print(f'Number of rows: {num_rows}')
print(f'Number of columns: {num_columns}')

#3/ Print the name of the columns.
print("==========Cau 3==========")
print(data.columns.tolist())

#4/ What is the type of the column names?
print("==========Cau 4==========")
column_names_type = type(data.columns)
print(f'The type of the column names is: {column_names_type}')

#5/ Get the country column and save it to its own variable. Show the first 5 observations.
print("==========Cau 5==========")
country_colum = data["country"]
print(country_colum.head(5))

#6/ Show the last 5 observations of this column.
print("==========Cau 6==========")
print(data.tail(5))

#7/ Look at country, continent and year. Show the first 5 observations of these columns, and the last 5 observations.
print("==========Cau 7==========")
looked = data[["country", "continent", "year"]]
print(looked.head(5))
print(looked.tail(5))

#8/ How to get the first row of tsv file? How to get the 100th row.
print("==========Cau 8==========")
print(data.head(1))
print(data.iloc[99])

#9/ Try to get the first column by using a integer index. And get the first and last column by passing the integer index.
print("==========Cau 9==========")
first_column = data.iloc[:, 0]
first_and_last_columns = data.iloc[:, [0, -1]]
print("First Column:")
print(first_column)
print("\nFirst and Last Columns:")
print(first_and_last_columns)

#10/ How to get the last row with .loc? Try with index -1? Correct?
print("==========cau 10==========")
last_row_label = data.index[-1]
last_row = data.loc[last_row_label]
print("Last Row:")
print(last_row)

#11/ How to select the first, 100th, 1000th rows by two methods?
print("==========Cau 11==========")
first_row_iloc = data.iloc[0]
hundredth_row_iloc = data.iloc[99]
thousandth_row_iloc = data.iloc[999]

first_row_loc = data.loc[data.index[0]]
hundredth_row_loc = data.loc[data.index[99]]
thousandth_row_loc = data.loc[data.index[999]]

print("========Using iloc===========")
print("First Row:")
print(first_row_iloc)
print("\n100th Row:")
print(hundredth_row_iloc)
print("\n1000th Row:")
print(thousandth_row_iloc)

print("\n==========Using loc=========")
print("First Row:")
print(first_row_loc)
print("\n100th Row:")
print(hundredth_row_loc)
print("\n1000th Row:")
print(thousandth_row_loc)

#12/ Get the 43rd country in our data using .loc, .iloc?
print("==========Cau 12==========")
print("=====iloc=====")
print(data.iloc[43])
print("\n=====loc=====")
print(data.loc[data.index[43]])

#13/ How to get the first, 100th, 1000th rows from the first, 4th and 6th columns?
print("==========Cau 13==========")
rows_to_select = [0, 99, 999]
columns_to_select = [0, 3, 5]
selected_data_iloc = data.iloc[rows_to_select, columns_to_select]
print(selected_data_iloc)

#14/ Get first 10 rows of our data (tsv file)?
print("==========Cau 14==========")
print(data.head(10))

#15/ For each year in our data, what was the average life expectation?
print("==========Cau 15==========")
average_life_expectancy = data.groupby('year')['lifeExp'].mean()
print(average_life_expectancy)

#16/ Using subsetting method for the solution of 15/?
print("==========Cau 16==========")
subset_data = data[['year', 'lifeExp']]
average_life_expectancy = subset_data.groupby('year')['lifeExp'].mean()
print(average_life_expectancy)

#17/ Create a series with index 0 for ‘banana’ and index 1 for ’42’?
print("==========Cau 17==========")
data = pd.Series(['banana', 42], index=[0, 1])
print(data)

#18/ Similar to 17, but change index ‘Person’ for ‘Wes MCKinney’ and index ‘Who’ for ‘Creator of Pandas’?
print("==========Cau 18==========")
data = pd.Series(['Wes McKinney', 'Creator of Pandas'], index=['Person', 'Who'])
print(data)

#19/ Create a dictionary for pandas with the data as ‘Occupation’: [’Chemist’, ’Statistician’],
# ’Born’: [’1920-07-25’, ’1876-06-13’],’Died’: [’1958-04-16’, ’1937-10-16’],’Age’: [37, 61] and the index is
# ‘Franklin’,’Gosset’ with four columns as indicated.
print("==========Cau 19==========")
data = {
    'Occupation': ['Chemist', 'Statistician'],
    'Born': ['1920-07-25', '1876-06-13'],
    'Died': ['1958-04-16', '1937-10-16'],
    'Age': [37, 61]
}
ds = pd.DataFrame(data, index=['Franklin', 'Gosset'])
print(ds)