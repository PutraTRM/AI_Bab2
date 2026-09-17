# Example 2.19

# Install terlebih dahulu jika belum ada:
# pip install pandas openpyxl

import pandas

df = pandas.read_excel('test.xlsx', sheet_name='Sheet1')

print(df)