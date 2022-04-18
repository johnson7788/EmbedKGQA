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
