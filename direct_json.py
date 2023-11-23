import pandas as pd
import json

# Assuming your Excel file is named 'your_dataset.xlsx' and is in the same directory as this script
excel_file = 'Fullset.xlsx'

# Read the Excel file into a pandas DataFrame
df = pd.read_excel(excel_file)

# Convert the DataFrame to a list of dictionaries in the specified format
intents_list = []
for index, row in df.iterrows():
    intent = {
        "tags": str(row['tags']).strip(),
        "patterns": [pattern.strip() for pattern in str(row['patterns']).split(',')],
        "responses": [response.strip() for response in str(row['responses']).split(',')]
    }
    intents_list.append(intent)

# Create a dictionary with the 'intents' key and the list of intents
json_data = {"intents": intents_list}

# Convert the dictionary to JSON and write it to a file named 'Bigdata.json'
with open('Fullset.json', 'w', encoding='utf-8') as json_file:
    json.dump(json_data, json_file, ensure_ascii=False, indent=2)

print("JSON file created successfully.")
