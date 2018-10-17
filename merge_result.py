#!/usr/bin/python
import os
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default="experiments/HAN/result/", help="Directory containing the dataset")

def merge(basedir, output_file):
  columns = "location_traffic_convenience,location_distance_from_business_district,location_easy_to_find,\
service_wait_time,service_waiters_attitude,service_parking_convenience,service_serving_speed,\
price_level,price_cost_effective,price_discount,\
environment_decoration,environment_noise,environment_space,environment_cleaness,\
dish_portion,dish_taste,dish_look,dish_recommendation,\
others_overall_experience,others_willing_to_consume_again".split(",")
  result = [pd.Series(name="content")]
  for column in columns:
    label = os.path.join(basedir, column + "_result.csv")
    data = pd.read_csv(label, encoding='utf-8', names=["id", column], header=None, skiprows=1)
    data.set_index('id')
    result.append(data[column].map(lambda e: int(e.strip("[]")) - 2))
  result_all = pd.concat(result, axis=1)
  output_file = os.path.join(basedir, output_file)
  result_all.to_csv(output_file, encoding='utf-8', index_label="id")


if __name__ == "__main__":
  args = parser.parse_args()
  merge(args.data_dir, "han_result.csv")

