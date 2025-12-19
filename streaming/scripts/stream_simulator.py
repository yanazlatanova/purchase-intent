"""
Real Streaming Data Source Simulator
Generates online shoppers data in real-time via socket stream
"""

import socket
import time
import pandas as pd
import json
import sys

def stream_data_from_csv(csv_path, host='0.0.0.0', port=9999, batch_size=50, delay=5):
    """
    Reads CSV and streams rows as real-time data via socket
    
    Args:
        csv_path: Path to online_shoppers_all.csv
        host: Socket host
        port: Socket port
        batch_size: Rows per batch
        delay: Seconds between batches
    """
    print(f"Starting Data Stream Server on {host}:{port}")
    print(f"Reading from: {csv_path}")
    print(f"Batch size: {batch_size} rows | Delay: {delay}s\n")
    
    # Load CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(df)} records from CSV")
    except FileNotFoundError:
        print(f"✗ CSV file not found: {csv_path}")
        sys.exit(1)
    
    # Create socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)
    
    print(f"✓ Socket server listening on {host}:{port}")
    print("Waiting for client connection...\n")
    
    # Accept client connection
    client_socket, client_address = server_socket.accept()
    print(f"✓ Client connected from {client_address}")
    print("Starting data stream...\n")
    
    try:
        batch_num = 0
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_num += 1
            
            # Convert rows to CSV format (one row per line)
            for idx, row in batch.iterrows():
                # Skip header for first row of each batch
                csv_line = ",".join(str(val) for val in row.values) + "\n"
                try:
                    client_socket.sendall(csv_line.encode('utf-8'))
                except:
                    print("✗ Client disconnected")
                    break
            
            print(f"Batch {batch_num}: Sent {len(batch)} records " +
                  f"(Total: {min(i + batch_size, len(df))}/{len(df)})")
            
            # Wait before sending next batch
            time.sleep(delay)
        
        print(f"\n✓ Stream completed! All {len(df)} records sent.")
        print("Keeping socket open... (Ctrl+C to close)")
        
        # Keep socket open
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n✓ Stream interrupted by user")
    finally:
        client_socket.close()
        server_socket.close()
        print("✓ Socket closed")

if __name__ == "__main__":
    # Configuration
    CSV_PATH = "/opt/spark/data/online_shoppers_all.csv"
    HOST = "0.0.0.0"  # Listen on all interfaces
    PORT = 9999
    BATCH_SIZE = 30   # Stream 50 rows at a time
    DELAY = 5         # 5 seconds between batches
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        BATCH_SIZE = int(sys.argv[1])
    if len(sys.argv) > 2:
        DELAY = int(sys.argv[2])
    
    stream_data_from_csv(CSV_PATH, HOST, PORT, BATCH_SIZE, DELAY)