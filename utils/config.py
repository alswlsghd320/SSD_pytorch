import configparser
from time import strftime

def config_read(config_path='config/config.ini'):
    # 설정파일 읽기
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')

    return config