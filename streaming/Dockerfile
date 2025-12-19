FROM apache/spark:4.0.1

# Switch to root user for installation
USER root

RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Update pip and install compatible versions + additional dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.0.3 \
    pyarrow==12.0.1 \
    requests==2.31.0 \
    pyspark==3.5.0 \
    scikit-learn==1.3.0

# Switch back to spark user
USER spark

# Set working directory
WORKDIR /opt/spark

# Create necessary directories
RUN mkdir -p /opt/spark/data /opt/spark/checkpoints /opt/spark/output
