
# Create data directory if it doesn't exist
mkdir -p data

# Download yellow taxi data for 2025-01 if it doesn't exist
if [ ! -f "data/yellow_tripdata_2025-01.parquet" ]; then
    echo "Downloading yellow taxi data for 2025-01..."
    curl -o data/yellow_tripdata_2025-01.parquet https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet
else
    echo "File data/yellow_tripdata_2025-01.parquet already exists, skipping download."
fi

# Download yellow taxi data for 2025-02 if it doesn't exist
if [ ! -f "data/yellow_tripdata_2025-02.parquet" ]; then
    echo "Downloading yellow taxi data for 2025-02..."
    curl -o data/yellow_tripdata_2025-02.parquet https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-02.parquet
else
    echo "File data/yellow_tripdata_2025-02.parquet already exists, skipping download."
fi

# Download yellow taxi data for 2025-03 if it doesn't exist
if [ ! -f "data/yellow_tripdata_2025-03.parquet" ]; then
    echo "Downloading yellow taxi data for 2025-03..."
    curl -o data/yellow_tripdata_2025-03.parquet https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-03.parquet
else
    echo "File data/yellow_tripdata_2025-03.parquet already exists, skipping download."
fi