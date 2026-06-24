import pandas as pd

# Assuming your Excel file is named 'your_file.xlsx' and the column name is 'cardio'
file_path = '/Users/kao900531/Desktop/Cardio.xlsx'

# Read the Excel file into a DataFrame
df = pd.read_excel(file_path)

# Map values in the 'cardio' column to 'Having CVD' and 'No CVD'
df['cardio'] = df['cardio'].map({1: 'Having CVD', 0: 'No CVD'})

# Print the updated DataFrame
print(df)

# Save the updated DataFrame to a new Excel file
output_file_path = '/Users/kao900531/Desktop/Cardio_1.xlsx'
df.to_excel(output_file_path, index=False)
