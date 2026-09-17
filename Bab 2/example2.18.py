# Example 2.18

# Install terlebih dahulu jika belum ada:
# pip install xlsxwriter

import pandas as pd

data = {
    'Name': ['Tony', 'Robert', 'John', 'Alice'],
    'Age': [18, 24, 19, 21]
}

df = pd.DataFrame(data, columns=['Name', 'Age'])

print(df)

writer = pd.ExcelWriter("test.xlsx", engine='xlsxwriter')

# Menulis DataFrame ke Excel
df.to_excel(
    writer,
    sheet_name='Sheet1',
    index=False
)

writer.close()