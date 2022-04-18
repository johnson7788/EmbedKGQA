#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2022/4/18 11:45 上午
# @File  : test_api.py
# @Author:
# @Desc  : 测试
import unittest
import requests
import time, os
import json
import base64
import random
import string
import pickle
import sys

class LSTMKQGATestCase(unittest.TestCase):
    host_server = f'http://l8:9966'
    def test_lstmkgqa_file(self):
        """
        测试文件接口
        :return:
        :rtype:
        """
        url = f"{self.host_server}/api/predict_file"
        params = {'data_apth': "./../data/QA_data/MetaQA/qa_test_1hop.txt"}
        headers = {'content-type': 'application/json'}
        r = requests.post(url, headers=headers, data=json.dumps(params), timeout=360)
        result = r.json()
        print(result)
        assert r.status_code == 200
        assert result is not None, "返回结果为None"
        #检查结果，里面肯定是字典格式
        print("对文件接口测试完成")
    def test_lstmkgqa(self):
        """
        测试数据的正确答案
        what does [Grégoire Colin] appear in	Before the Rain
        [Joe Thomas] appears in which movies	The Inbetweeners Movie|The Inbetweeners 2
        what films did [Michelle Trachtenberg] star in	Inspector Gadget|Black Christmas|Ice Princess|Harriet the Spy|The Scribbler
        what does [Helen Mack] star in	The Son of Kong|Kiss and Make-Up|Divorce
        测试接口
        :return:
        :rtype:
        """
        url = f"{self.host_server}/api/predict"
        data = [
            ['Grégoire Colin', 'what does NE appear in'],
            ['Joe Thomas', 'NE appears in which movies'],
            ['Michelle Trachtenberg', 'what films did NE star in'],
            ['Helen Mack', 'what does NE star in'],
            ['Shahid Kapoor', 'what films did NE act in'],
        ]
        params = {'data':data}
        headers = {'content-type': 'application/json'}
        r = requests.post(url, headers=headers, data=json.dumps(params), timeout=360)
        result = r.json()
        print(result)
        assert r.status_code == 200
        assert result is not None, "返回结果为None"
        #检查结果，里面肯定是字典格式
        print("对文件接口测试完成")
