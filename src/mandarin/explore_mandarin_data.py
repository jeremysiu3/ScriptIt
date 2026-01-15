import zipfile
import pandas as pd

zip_path = "data/raw/mandarin_news_dataset.zip"

print("Exploring Chinese news dataset...")

# Open zip file
with zipfile.ZipFile(zip_path, 'r') as zip_file:
    # List all files in the zip
    file_list = zip_file.namelist()
    
    print(f"\nFiles in zip: {len(file_list)}")
    for f in file_list:
        print(f"  - {f}")
    
    # Read first CSV to see structure
    if file_list:
        first_csv = [f for f in file_list if f.endswith('.csv')][0]
        print(f"\nReading {first_csv}...")
        
        with zip_file.open(first_csv) as csv_file:
            df = pd.read_csv(csv_file, nrows=5)  # Just first 5 rows
            
            print(f"\nColumns: {list(df.columns)}")
            print(f"\nFirst few rows:")
            print(df.head())
            
            # Check if there's Chinese text
            for col in df.columns:
                sample = str(df[col].iloc[0])[:100]
                print(f"\n{col}: {sample}...")