#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
from zhon import cedict
from zhon import hanzi
import string
import csv
import re
import commands

stop_char = set()
def load_stop_char(char_file):
  global stop_char
  fhandler = open(char_file)
  for line in fhandler:
    c = line.strip('\r\n')
    stop_char.add(c)
  fhandler.close()
  stop_char.add('\r')
  stop_char.add('\n')
  ctrl = [8204, 8205, 8206, 8207, 8234, 8236, 8237, 8238, 8298, 8299, 8300, 8301, 8302, 8303]
  ctrl += range(0, 32)
  for c in ctrl:
    stop_char.add(unichr(c))

def remove_chinese_punctuation(string):
  #return re.sub(ur"[%s]+" %punctuation, u" ", string.decode("utf-8")).encode("utf-8") # 需要将str转换为unicode
  return re.sub(ur"[%s]+" %hanzi.punctuation, u" ", string)

def filter_stop_char(string):
  string = string.replace('\n', ' ')
  string = remove_chinese_punctuation(string)
  #return string
  global stop_char
  sentence = ''
  prev_space = False
  for char in string:
    if char in stop_char:
      continue
    elif char != ' ':
      sentence += char
      prev_space = False
    elif prev_space:
      continue
    else:
      sentence += char
      prev_space = True
  return sentence

def transform(char):
  if char.isalnum():
    return char.lower()
  if char in cedict.simplified:
    return char
  if char == "." or char in hanzi.stops:
    return "."
  if char.isspace() or char in hanzi.non_stops or char in string.punctuation:
    return " "
  if char == u"、":
    return " "
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

def append_content_ws(input_file, content_file, ws_file, output_file):
  data = pd.read_csv(input_file, encoding='utf-8')
  data.drop(columns=['content'], inplace=True)
  content = pd.read_csv(content_file, names=['content'], encoding='utf-8')
  content_ws = pd.read_csv(ws_file, names=['content_ws'], encoding='utf-8')
  result = pd.concat([content, data, content_ws], axis=1)
  result.to_csv(output_file, quoting=csv.QUOTE_NONNUMERIC, index=False, encoding='utf-8')

if '__main__' == __name__:
  #load_stop_char("stop_char.txt")
  #print len(stop_char)
  test_file = 'raw_data/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv'
  train_file = 'raw_data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv'
  valid_file = 'raw_data/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv'
  #get_content(test_file, 'data/test_content.txt')
  #get_content(train_file, 'data/train_content.txt')
  #get_content(valid_file, 'data/valid_content.txt')

  commands.getoutput("./segment.sh data/test_content.txt > data/test_content_words.txt")
  commands.getoutput("./segment.sh data/train_content.txt > data/train_content_words.txt")
  commands.getoutput("./segment.sh data/valid_content.txt > data/valid_content_words.txt")

  append_content_ws(test_file, 'data/test_content.txt', 'data/test_content_words.txt', 'data/testa.csv')
  append_content_ws(train_file, 'data/train_content.txt', 'data/train_content_words.txt', 'data/train.csv')
  append_content_ws(valid_file, 'data/valid_content.txt', 'data/valid_content_words.txt', 'data/valid.csv')
