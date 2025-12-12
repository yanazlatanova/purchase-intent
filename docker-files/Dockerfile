FROM apache/spark:4.0.1

# Switch to root to install system-wide
USER root

# Install pandas, pyarrow, and jupyter globally
RUN pip install --no-cache-dir "pandas==2.3.2" "pyarrow==21.0.0"

# Switch back to the default spark user
USER spark

# set $SPARK_CONF_DIR
ENV SPARK_CONF_DIR=/opt/spark/conf
COPY spark-defaults.conf /opt/spark/conf/spark-defaults.conf