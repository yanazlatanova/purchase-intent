import csv
import sys

def infer_type(value):
    """Infer the data type of a value"""
    if value == '' or value is None:
        return 'empty'
    
    try:
        int(value)
        return 'integer'
    except ValueError:
        pass
    
    try:
        float(value)
        return 'float'
    except ValueError:
        pass
    
    return 'string'

def read_csv_schema(file_path):
    """
    Read a CSV file and display its schema information
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            
            # Read header
            headers = next(csv_reader)
            
            # Initialize data structures
            column_types = {col: {} for col in headers}
            column_samples = {col: [] for col in headers}
            row_count = 0
            
            # Read all rows
            for row in csv_reader:
                row_count += 1
                for i, value in enumerate(row):
                    if i < len(headers):
                        col_name = headers[i]
                        value_type = infer_type(value)
                        
                        # Count types
                        if value_type in column_types[col_name]:
                            column_types[col_name][value_type] += 1
                        else:
                            column_types[col_name][value_type] = 1
                        
                        # Store sample values (first 3)
                        if len(column_samples[col_name]) < 3:
                            column_samples[col_name].append(value)
            
            # Display schema
            print(f"=== CSV Schema for: {file_path} ===\n")
            print(f"Total Rows: {row_count}")
            print(f"Total Columns: {len(headers)}\n")
            
            print("Column Details:")
            print("-" * 100)
            print(f"{'Column Name':<25} {'Inferred Type':<15} {'Empty Values':<15} {'Sample Values'}")
            print("-" * 100)
            
            for col in headers:
                types = column_types[col]
                
                # Determine primary type
                main_type = 'string'
                if 'integer' in types and types['integer'] > row_count * 0.8:
                    main_type = 'integer'
                elif 'float' in types and types['float'] > row_count * 0.8:
                    main_type = 'float'
                
                empty_count = types.get('empty', 0)
                samples = ', '.join(column_samples[col][:3])
                if len(samples) > 40:
                    samples = samples[:37] + '...'
                
                print(f"{col:<25} {main_type:<15} {empty_count:<15} {samples}")
            
            print("\n" + "=" * 100)
            
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error reading CSV: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python csv_schema_reader.py <path_to_csv_file>")
        print("\nExample: python csv_schema_reader.py data.csv")
    else:
        read_csv_schema(sys.argv[1])