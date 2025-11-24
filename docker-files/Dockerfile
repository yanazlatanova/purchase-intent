FROM apache/spark:4.0.1

# Switch to root to install system-wide
USER root

# Install pandas, pyarrow, and jupyter globally
RUN pip install --no-cache-dir "pandas==2.3.2" "pyarrow==21.0.0"

# Create directory for notebooks
RUN mkdir -p /opt/spark/notebooks

# Switch back to the default spark user
USER spark