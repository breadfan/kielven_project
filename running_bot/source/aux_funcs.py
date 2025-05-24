import math
import requests
from datetime import date, timedelta
from typing import List
from PIL import Image
from io import BytesIO
import pandas as pd
import time
import os
from functools import wraps
import yaml

"""
 Вспомогательные функции для обращения к серверам wb с целью получения данных о товаре.
"""
 
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

MPSTATS_TOKEN = config['mpstats_token']
DATA_FOLDER = config['data_folder']
DEVELOPER_CHAT_ID =  config['developer_chat_id'] 
# декоратор для измерения времени работы функции, так же показывает какие аргументы она приняла
# если много аргументов, то лучше удалить их 
def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        report_text = f'Function {func.__name__}{args} {kwargs} took {total_time:.4f} seconds'
        make_tg_report(report_text)
        # print(report_text)
        return result
    return timeit_wrapper

def make_tg_report(text) -> None:
    token = ''
    method = 'sendMessage'
    chat_id = 11111111

    _ = requests.post(
            url='https://api.telegram.org/bot{0}/{1}'.format(token, method),
            data={'chat_id': chat_id, 'text': text}
        ).json()