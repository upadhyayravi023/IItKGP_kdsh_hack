#######################-----Data keepers------###############################
#This code is to merge the two csv files

import pandas as pd


df1 = pd.read_csv('predictions.csv')  
df2 = pd.read_csv('conference_recommendations.csv')  

print("Columns in df1:", df1.columns)
print("Columns in df2:", df2.columns)

# Merge the two DataFrames on the 'file_name' column
combined_df = pd.merge(df1, df2, on='file_name', how='inner')  
combined_df.to_csv('combined_output.csv', index=False)

print("Combined CSV saved to 'combined_output.csv'.")

