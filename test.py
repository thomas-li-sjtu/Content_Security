# !/usr/bin/env python3
# -*- coding utf-8 -*-
# @TIME： 2020/10/20   12:45
# @FILE： test.py
# @IDE ： PyCharm
# @contact: 980226547@qq.com
import re

p = re.compile(r'\w')
result = p.sub(r"\w", "abc")
print(result)