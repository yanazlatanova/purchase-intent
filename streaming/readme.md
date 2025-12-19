docker exec spark-app spark-submit  --master spark://spark-master:7077 --deploy-mode client /opt/spark/work-dir/SGD_streaming.py

docker exec stream-simulator python stream_simulator.py