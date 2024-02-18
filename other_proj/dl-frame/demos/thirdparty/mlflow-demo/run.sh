ps aux | grep "mlflow"| awk '{print $2}'| xargs kill -9 

nohup mlflow ui &


mlflow run ./ -P alpha=0.4 --no-conda > train.log  2>&1

run_id=`tail -n 1 train.log|awk -F"'" '{print $2}'`

nohup mlflow models serve -m runs:/$run_id/model --port 3333 --no-conda &

sleep 10

curl -d '{"columns":["fixed acidity","volatile acidity"], "data":[[1,1]]}' -H 'Content-Type: application/json; format=pandas-split' -X POST localhost:3333/invocations


