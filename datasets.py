import os
from dotenv import load_dotenv

load_dotenv()
CTU_DIR = os.getenv('CTU_DIR')
NCC_DIR = os.getenv('NCC_DIR')
NCC_2_DIR = os.getenv('NCC_2_DIR')

list_dataset = [
  CTU_DIR+'/1/capture20110810.binetflow',
  CTU_DIR+'/2/capture20110811.binetflow',
  CTU_DIR+'/3/capture20110812.binetflow',
  CTU_DIR+'/4/capture20110815.binetflow',
  CTU_DIR+'/5/capture20110815-2.binetflow',
  CTU_DIR+'/6/capture20110816.binetflow',
  CTU_DIR+'/7/capture20110816-2.binetflow',
  CTU_DIR+'/8/capture20110816-3.binetflow',
  CTU_DIR+'/9/capture20110817.binetflow',
  CTU_DIR+'/10/capture20110818.binetflow',
  CTU_DIR+'/11/capture20110818-2.binetflow',
  CTU_DIR+'/12/capture20110819.binetflow',
  CTU_DIR+'/13/capture20110815-3.binetflow',
  
  NCC_DIR+'/scenario_dataset_1/dataset_result.binetflow',
  NCC_DIR+'/scenario_dataset_2/dataset_result.binetflow',
  NCC_DIR+'/scenario_dataset_3/dataset_result.binetflow',
  NCC_DIR+'/scenario_dataset_4/dataset_result.binetflow',
  NCC_DIR+'/scenario_dataset_5/dataset_result.binetflow',
  NCC_DIR+'/scenario_dataset_6/dataset_result.binetflow',
  NCC_DIR+'/scenario_dataset_7/dataset_result.binetflow',
  NCC_DIR+'/scenario_dataset_8/dataset_result.binetflow',
  NCC_DIR+'/scenario_dataset_9/dataset_result.binetflow',
  NCC_DIR+'/scenario_dataset_10/dataset_result.binetflow',
  NCC_DIR+'/scenario_dataset_11/dataset_result.binetflow',
  NCC_DIR+'/scenario_dataset_12/dataset_result.binetflow',
  NCC_DIR+'/scenario_dataset_13/dataset_result.binetflow',

  NCC_2_DIR+'/sensor1/sensor1.binetflow',
  NCC_2_DIR+'/sensor2/sensor2.binetflow',
  NCC_2_DIR+'/sensor3/sensor3.binetflow',
]