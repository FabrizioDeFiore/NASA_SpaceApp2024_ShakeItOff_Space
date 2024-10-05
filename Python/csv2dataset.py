import os
import pandas as pd

source_folder = 'downsampled_signals_and_sampels'
target_folder = 'downsampled_signals_and_sampels'

for filename in os.listdir(source_folder):
    if filename.endswith('.csv'):
        source_file_path = os.path.join(source_folder, filename)
        target_file_path = os.path.join(target_folder, filename)
        
        # Read the source CSV file
        source_df = pd.read_csv(source_file_path)
        
        # Check if 'velocity(m/s)' column exists in the source file
        if 'velocity(m/s)' in source_df.columns:
            velocity_data = source_df['velocity(m/s)']
            
            # Read the target CSV file
            if os.path.exists(target_file_path):
                target_df = pd.read_csv(target_file_path)
                
                # Drop the first column
                target_df.drop(target_df.columns[0], axis=1, inplace=True)
                
                # Ensure only 'velocity(m/s)' and 'label' columns remain
                if 'label' in target_df.columns:
                    target_df = target_df[['velocity(m/s)', 'label']]
                else:
                    target_df['velocity(m/s)'] = velocity_data
                    target_df = target_df[['velocity(m/s)']]
                
                # Reverse the positions of the columns
                target_df = target_df[['velocity(m/s)', 'label']]
                
                # Save the updated DataFrame back to the target file
                target_df.to_csv(target_file_path, index=False)
                print(f"Target file {target_file_path} edited.")
            else:
                print(f"Target file {target_file_path} does not exist.")
        else:
            print(f"'velocity(m/s)' column not found in {source_file_path}")
    else:
        print(f"File {filename} is not a CSV file.")