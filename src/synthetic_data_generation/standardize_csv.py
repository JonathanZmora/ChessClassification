import pandas as pd
import os
import argparse


def expand_dataset(input_csv, output_csv):
    if not os.path.exists(input_csv):
        print(f"Error: Could not find {input_csv}")
        return

    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # List to hold the new expanded rows
    new_rows = []
    
    # Iterate through the existing rows
    for row in df.to_dict('records'):
        
        # Common metadata to keep for all 3 variants
        base_data = {
            'game_number': row['game_number'],
            'frame_number': row['frame_number'],
            'view': row['view'], 
            'fen': row['FEN']    
        }
        
        # Overhead 
        if pd.notna(row.get('warped_overhead_name')):
            entry = base_data.copy()
            entry['image_name'] = row['warped_overhead_name']
            entry['warp_type'] = 'overhead' # Optional metadata
            new_rows.append(entry)
            
        # West 
        if pd.notna(row.get('warped_west_name')):
            entry = base_data.copy()
            entry['image_name'] = row['warped_west_name']
            entry['warp_type'] = 'west'
            new_rows.append(entry)
            
        # East 
        if pd.notna(row.get('warped_east_name')):
            entry = base_data.copy()
            entry['image_name'] = row['warped_east_name']
            entry['warp_type'] = 'east'
            new_rows.append(entry)

    # Create new DataFrame
    expanded_df = pd.DataFrame(new_rows)
    
    # Save
    expanded_df.to_csv(output_csv, index=False)
    
    print("-" * 30)
    print(f"Conversion Complete!")
    print(f"Original Rows: {len(df)}")
    print(f"Expanded Rows: {len(expanded_df)}")
    print(f"Saved to:      {output_csv}")
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the input csv file")
    parser.add_argument("--output", required=True, help="Path to the output csv file")
    args = parser.parse_args()

    expand_dataset(args.input, args.output)