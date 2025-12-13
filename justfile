[working-directory: 'docker-files']
@start-spark:
  docker compose down; docker compose up -d

[working-directory: 'docker-files']
@submit-workers-jobs:
  (parallel --tag docker exec -i docker-files-spark-worker-{}-1 /opt/spark/bin/spark-submit /opt/spark/work-dir/worker{}.py ::: 1 2) > ../src/part2/job_output.log

[working-directory: 'docker-files']
@submit-master-job:
  (docker exec -i docker-files-spark-master-1 /opt/spark/bin/spark-submit /opt/spark/work-dir/joined.py) > ../src/part2/joined_output.log

@run-setup:
  uv run src/setup.py

[working-directory: 'docker-files']
@lazydocker:
  lazydocker