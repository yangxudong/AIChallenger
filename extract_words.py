#!/usr/bin/python
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import pandas as pd
from collections import Counter
import math
import itertools
import operator

wordCounter = Counter()
labelCounter = Counter()
labelWordCounter = {}

columns = "location_traffic_convenience,location_distance_from_business_district,location_easy_to_find,\
service_wait_time,service_waiters_attitude,service_parking_convenience,service_serving_speed,\
price_level,price_cost_effective,price_discount,\
environment_decoration,environment_noise,environment_space,environment_cleaness,\
dish_portion,dish_taste,dish_look,dish_recommendation,\
others_overall_experience,others_willing_to_consume_again".split(",")

def update_counter(row):
    wordCounter.update(row["words"])
    for c in columns:
        label = c + "_" + str(row[c])
        labelCounter[label] += 1
        cnter = labelWordCounter.setdefault(label, Counter())
        cnter.update(row["words"])

def statistic(input_files):
  print("reading input files...")
  dfs = [pd.read_csv(f, encoding='utf-8') for f in input_files]
  data = pd.concat(dfs, copy=False)
  print("transforming contents...")
  words = data.content_ws.str.split(" ")
  words = words.map(lambda arr: set([x for x in arr if x != ""]))
  data["words"] = words
  print("collecting statistical data...")
  data.apply(lambda row: update_counter(row), axis=1)
  N = len(data)
  print("data size: " + str(N))
  print("computing pmi scores...")
  classes = ["_-2", "_-1", "_0", "_1"]
  for column,clazz in itertools.product(columns, classes):
    label = column + clazz
    pmi_score = get_pmi_score(label, N)
    scores = sorted(pmi_score.items(), key=operator.itemgetter(1), reverse=True)
    print("writing pmi scores for " + label)
    output = open('data/'+label, 'w')
    for w, s in scores:
      output.write("{}\t{}\n".format(w, s))
    output.close()

def get_pmi_score(label, N):
  coocurrence = labelWordCounter[label]
  pmi = {}
  labelCnt = labelCounter[label]
  if labelCnt <= 0:
    return pmi
  for w, cnt in coocurrence.items():
    score = math.log((N * cnt / float(wordCounter[w]) / labelCnt), 2)
    pmi[w] = score
  return pmi

def load_dict(input_file):
  pmi = {}
  fileObj = open(input_file)
  for line in fileObj:
    kv = line.rstrip().split("\t")
    if len(kv) < 2: continue
    pmi[kv[0]] = float(kv[1])
  fileObj.close()
  return pmi

def write_dict(positive, negative, output_file):
  result = {}
  for word, score in positive.iteritems():
    neg_score = negative.get(word, 0.0)
    final_score = score - neg_score
    result[word] = final_score
  sorted_result = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
  print("writing to " + output_file)
  output = open(output_file, 'w')
  for word, score in sorted_result:
    output.write("{}\t{}\n".format(word, score))
  output.close()

def find_sentiment_words():
  data_dir = "data/"
  for column in columns:
    positive = load_dict(data_dir + column + "_1")
    negative = load_dict(data_dir + column + "_-1")
    write_dict(positive, negative, data_dir + column + "_positive")
    write_dict(negative, positive, data_dir + column + "_negative")
    

if '__main__' == __name__:
  #statistic(['data/train.csv', 'data/valid.csv'])
  find_sentiment_words()
