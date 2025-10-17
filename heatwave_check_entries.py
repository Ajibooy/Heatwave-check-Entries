import os
import pandas as pd
# --- Config ---
data_dir = 'C:/Users/Aina Ajibola/Desktop/New folder'  # Ensure the path is correct
file_path = os.path.join(data_dir, 'hot5_temperature_baku_1940_2025.csv')  # Path to CSV file
# --- Load the CSV data ---
temperature_data = pd.read_csv(file_path)
# Check the first few rows to verify the data is loaded correctly
print(temperature_data.head())
# Convert 'Date' column to datetime
temperature_data['Date'] = pd.to_datetime(temperature_data['Date'])
# Filter for summer months (June, July, August)
summer_data = temperature_data[temperature_data['Date'].dt.month.isin([6, 7, 8])]
# Check if the filtering by summer months worked
print(f"Filtered summer data (first 5 rows): \n{summer_data.head()}")
# Add a column for year extracted from 'Date'
summer_data['Year'] = summer_data['Date'].dt.year
# Apply the reference P90 value of 28.72°C to the summer data and check if temperatures exceed it
summer_data['Exceeds_P90'] = summer_data['Temperature (°C)'] > 28.72
# Check the summer data to see if the 'Exceeds_P90' column was created correctly
print(f"Summer data with P90 check: \n{summer_data[['Date', 'Temperature (°C)', 'Exceeds_P90']].head()}")
# Identify consecutive days exceeding P90 using a group ID
summer_data['Consecutive_Group'] = (summer_data['Exceeds_P90'] != summer_data['Exceeds_P90'].shift()).cumsum()
# Check the consecutive grouping
print(f"Summer data with consecutive groups: \n{summer_data[['Date', 'Year', 'Exceeds_P90', 'Consecutive_Group']].head()}")
# Filter for consecutive days exceeding P90 (3 or more consecutive days)
heatwaves = summer_data[summer_data['Exceeds_P90']].groupby(['Year', 'Consecutive_Group']).filter(lambda x: len(x) >= 3)
# Check if heatwaves were correctly filtered
print(f"Filtered heatwaves: \n{heatwaves[['Year', 'Date', 'Consecutive_Group']].head()}")
# Get the first heatwave of each year, ensuring only the earliest occurrence
first_heatwave_per_year = heatwaves.groupby('Year').agg({'Date': 'min'}).reset_index()
# Display the result with the year and the first heatwave start date
print(f"First heatwave per year: \n{first_heatwave_per_year[['Year', 'Date']]}")