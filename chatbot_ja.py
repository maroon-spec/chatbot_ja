import streamlit as st 
import numpy as np 
from PIL import Image
import base64

import os
import requests
import numpy as np
import pandas as pd
import json

st.header('Databricks Q&A bot')
st.write('''
- [カスタマーサービスとサポートにおける大規模言語モデルの革命をドライブする \- Qiita](https://qiita.com/taka_yayoi/items/447ab95af2b8493a04dd)
''')

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(question):
  token = st.secrets["DATABRICKS_TOKEN"]
  url = 'https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints/llm-qabot-endpoint-jmaru-jpn/invocations'
  headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
  dataset = pd.DataFrame({'question':[question]})
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')

  return response.json()

question = st.text_input("質問")
#question = pd.DataFrame({'question':[f"{qa}"]})

if question != "":
  response = score_model(question)

  answer = response['predictions'][0]["answer"]
  source = response['predictions'][0]["source"]

  st.write(f"回答: {answer}")
  st.write(f"ソース: [{source}]({source})")