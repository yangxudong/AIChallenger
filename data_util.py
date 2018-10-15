#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
from zhon import cedict
from zhon import hanzi
import string
import csv
import commands

en_stops = set(list(u".?!\r\n~;；丶ˋ～…"))
def transform(char):
  if char in en_stops or char in hanzi.stops:
    return "."
  if char.isalnum():
    return char.lower()
  if char in cedict.simplified:
    return char
  if char.isspace() or char in hanzi.non_stops or char in string.punctuation:
    return ";"
  if char == u"、":
    return ";"
  index = cedict.traditional.find(char)
  if index >= 0:
    return cedict.simplified[index]
  return ""

def translate(sentence):
  return "".join(map(transform, list(sentence)))

def get_content(input_file, output_file):
  data = pd.read_csv(input_file, encoding='utf-8')
  content = data.content.map(translate)
  content.to_csv(output_file, index=False, encoding='utf-8')

def get_clauses(sentence):
  result = []
  clauses = sentence.split(";")
  sent = []
  i = 0
  for c in clauses:
    words = c.split()
    if len(words) + len(sent) <= 100 and i < 8: # 每句话最多8个短句, 总长度限制在100以内
      sent.extend(words)
      i += 1
    else:
      if len(sent) > 0:
        result.append(sent)
      if len(words) > 100:
        for i in range(0, len(words), 100):
          result.append(words[i : i + 100])
        sent = []
        i = 0
      else:
        sent = words
        i = 1
  if len(sent) > 0:
    result.append(sent)
  return result

def doc_to_sentences(doc):
  sentences = doc.split(".")
  result = []
  for s in sentences:
    if not s: continue
    s = s.strip(";")
    if not s: continue
    clauses = get_clauses(s)
    if not clauses: continue
    first_clause_len = len(clauses[0])
    if len(result) > 0 and len(clauses) == 1 and first_clause_len < 10 and len(result[-1]) + first_clause_len <= 40:
      result[-1] += clauses[0]
    else:
      result.extend(clauses)
  pad_result, sent_len = pad_sequence(result, 100, "<pad>")
  return "\n".join([" ".join(row) for row in pad_result]), ":".join([str(l) for l in sent_len])

def pad_sequence(inputs, maxlen, value):
  sent_len = []
  for i in range(len(inputs)):
    s = inputs[i]
    length = len(s)
    if length > maxlen:
      inputs[i] = s[:maxlen]
      sent_len.append(maxlen)
    elif len(s) < maxlen:
      sent_len.append(length)
      inputs[i].extend([value] * (maxlen - len(s)))
    else:
      sent_len.append(length)
  return inputs, sent_len


def append_content_ws(input_file, content_file, ws_file, output_file, shuffle=True):
  data = pd.read_csv(input_file, encoding='utf-8')
  #data.drop(columns=['content'], inplace=True)
  formatted_content = pd.read_csv(content_file, names=['formatted_content'], encoding='utf-8')
  result = pd.concat([data, formatted_content], axis=1)
  content_ws = pd.read_csv(ws_file, names=['content_ws'], encoding='utf-8')
  result["sentences"], result["sentence_len"] = zip(*content_ws.content_ws.map(doc_to_sentences))
  if shuffle:
    result = result.sample(frac=1)
    result.to_csv(output_file, quoting=csv.QUOTE_NONNUMERIC, index=False, encoding='utf-8')
  else:
    result.to_csv(output_file, index=False, encoding='utf-8')

def category_count(input_file, output_file):
  data = pd.read_csv(input_file, encoding='utf-8')
  columns = "location_traffic_convenience,location_distance_from_business_district,location_easy_to_find,\
service_wait_time,service_waiters_attitude,service_parking_convenience,service_serving_speed,\
price_level,price_cost_effective,price_discount,\
environment_decoration,environment_noise,environment_space,environment_cleaness,\
dish_portion,dish_taste,dish_look,dish_recommendation,\
others_overall_experience,others_willing_to_consume_again".split(",")
  column_value_count = [ data[x].value_counts() for x in columns ]
  output = pd.concat(column_value_count, axis=1)
  output.T.to_csv(output_file, sep='|')


if '__main__' == __name__:
  test_file = 'raw_data/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv'
  train_file = 'raw_data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv'
  valid_file = 'raw_data/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv'

  get_content(test_file, 'data/test_content.txt')
  get_content(train_file, 'data/train_content.txt')
  get_content(valid_file, 'data/valid_content.txt')

  commands.getoutput("./segment.sh data/test_content.txt > data/test_content_words.txt")
  commands.getoutput("./segment.sh data/train_content.txt > data/train_content_words.txt")
  commands.getoutput("./segment.sh data/valid_content.txt > data/valid_content_words.txt")

  append_content_ws(test_file, 'data/test_content.txt', 'data/test_content_words.txt', 'data/testa.csv', shuffle=False)
  append_content_ws(train_file, 'data/train_content.txt', 'data/train_content_words.txt', 'data/train.csv')
  append_content_ws(valid_file, 'data/valid_content.txt', 'data/valid_content_words.txt', 'data/valid.csv')

  #category_count( 'data/train.csv', 'data/value_count.csv')
