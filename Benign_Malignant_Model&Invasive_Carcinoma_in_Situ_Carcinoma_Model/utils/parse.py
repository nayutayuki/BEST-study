'''
@Date: 2019-12-05 04:23:57
@Author: Yong Pi
@LastEditors: Yong Pi
@LastEditTime: 2019-12-05 04:23:57
@Description: All rights reserved.
'''
import json
import ruamel.yaml


def parse_yaml(file='cfgs/default.yaml'):
    with open(file) as f:
        return ruamel.yaml.load(f, Loader=ruamel.yaml.Loader)


def format_config(config, indent=2):
    return json.dumps(config, indent=indent)
