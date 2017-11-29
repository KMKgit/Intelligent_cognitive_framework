/*-- sudo -s --*/

1. mongod

2. (while [ 1 ]; do sleep 1; done) | python py/energy_watt_service/backend_request_energy_watt_service.py NNpZyj5GxvHo1PBP train_201611_21_05_59
   => console "loaded model" 기다리기

4. node auth_server.js

5. node server.js


ip     : 52.79.71.50:8080/run
apikey : NNpZyj5GxvHo1PBP
model  : train_201611_21_05_59

