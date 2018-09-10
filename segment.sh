#!/bin/bash
work_home=$(readlink -f $(dirname $0))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib

ALIWS_HOME=/home/xudong.yang/AliWS-1.4.0.0
#/home/xudong.yang/segment/segment -conf $ALIWS_HOME/conf/AliTokenizer.conf -ws O2O_CHN -tokentype 2 -input $1 -stopword /home/xudong.yang/segment/stopwords.txt
/home/xudong.yang/segment/segment -conf $ALIWS_HOME/conf/AliTokenizer.conf -ws O2O_CHN -tokentype 2 -input $1 -stopword /home/xudong.yang/segment/simple_stopwords.txt
