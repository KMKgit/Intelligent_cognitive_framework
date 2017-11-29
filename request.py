import requests
import json
headers = {'content-type' : 'application/json'}
data ={"eventName":"send_jandi_msg","data":{"value1":"test"}}
url = "https://maker.ifttt.com/use/bLcV00aD5jPLKD5fNTRQ97"

response = requests.post(url,data=json.dumps(data),headers=headers)
print response.status_code
print response.text

curl -X POST -H "Content-Type: application/json" -d '{"value1":"test","value2":"test2","value3":"test3"}' https://maker.ifttt.com/trigger/drop_box/with/key/bLcV00aD5jPLKD5fNTRQ97