#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: evan-gyy
import json
import re
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | [%(levelname)s] %(message)s')
logger = logging.getLogger()

rel2id = {"ART": 0, "GEN-AFF": 1, "ORG-AFF": 2, "PART-WHOLE": 3, "PER-SOC": 4, "PHYS": 5}
id2rel = {"0": "ART", "1": "GEN-AFF", "2": "ORG-AFF", "3": "PART-WHOLE", "4": "PER-SOC", "5": "PHYS"}

output_path = "RIFRE-main/datasets/data/hw/"


def format_file(file):
    """
    格式化train/dev文件
    """
    with open(file, 'r') as f:
        text = f.read()
    with open(file.replace('txt', "annotation.re"), 'r') as f:
        ann = []
        for lines in f.read().split('\n'):
            if lines:
                rel, e1, e2 = lines.split('\t')
                ann.append({
                    'relation': rel,
                    'h': e1.split(','),
                    't': e2.split(',')
                })
    sent = get_sentence(text)
    s_i = 0
    result = []
    for a_i in ann:
        h_start = int(a_i['h'][1])
        t_start = int(a_i['t'][1])
        tmp = sent[s_i]['pos']
        while h_start < tmp[0] or t_start < tmp[0] or h_start > tmp[-1] or t_start > tmp[-1]:
            s_i += 1
            if tmp[0] < h_start < tmp[-1] and t_start > tmp[-1] or tmp[0] < t_start < tmp[-1] and h_start > tmp[-1]:
                sent[s_i]['token'] = sent[s_i - 1]['token'] + sent[s_i]['token']
                sent[s_i]['pos'] = sent[s_i - 1]['pos'] + sent[s_i]['pos']
                tmp = sent[s_i]['pos']
                break
            else:
                tmp = sent[s_i]['pos']
        try:
            h_pos = tmp.index(h_start)
            t_pos = tmp.index(t_start)
        except ValueError:
            continue
        if len(sent[s_i]['token']) > 512:
            continue
        result.append({
            'file_name': file.replace('.txt', ''),
            'token': sent[s_i]['token'],
            'relation': a_i['relation'],
            'h': {'pos': [h_pos, h_pos + 1]},
            't': {'pos': [t_pos, t_pos + 1]}
        })
    return result


def format_test(test_path='test', result_path='Result'):
    """
    格式化test文件
    """
    logger.info(f'开始预处理: {test_path}')
    with open(result_path, 'r') as f:
        result = f.read().split('\n')
    result = [[i.split() for i in line.strip().split('\t')] for line in result]
    data = []
    for i, file in enumerate(os.listdir(test_path)):
        if result[i] == [[]]:
            continue
        with open(test_path + '/' + file, 'r') as f:
            text = f.read()
        sent = get_sentence(text)
        s_i = 0
        for a_i in result[i]:
            h_start = int(a_i[1])
            t_start = int(a_i[3])
            tmp = sent[s_i]['pos']
            while h_start < tmp[0] or t_start < tmp[0] or h_start > tmp[-1] or t_start > tmp[-1]:
                s_i += 1
                if tmp[0] < h_start < tmp[-1] and t_start > tmp[-1] or tmp[0] < t_start < tmp[-1] and h_start > tmp[-1]:
                    sent[s_i]['token'] = sent[s_i-1]['token'] + sent[s_i]['token']
                    sent[s_i]['pos'] = sent[s_i-1]['pos'] + sent[s_i]['pos']
                    tmp = sent[s_i]['pos']
                    break
                else:
                    tmp = sent[s_i]['pos']
            try:
                h_pos = tmp.index(h_start)
                t_pos = tmp.index(t_start)
            except ValueError:
                continue
            data.append({
                'file_name': file.replace('.txt', ''),
                'token': sent[s_i]['token'],
                'relation': id2rel[a_i[0]],
                'h': {'pos': [h_pos, h_pos + 1]},
                't': {'pos': [t_pos, t_pos + 1]},
                'pos': a_i[1: 5]
            })
    with open(output_path + 'test.json', 'w') as f:
        f.write(json.dumps(data))
    with open(output_path + 'test.txt', 'w') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')


def format_back(data):
    for i in data:
        h = i['token'][i['h']['pos'][0]]
        t = i['token'][i['t']['pos'][0]]
        print(i['relation'], h, t)


def get_sentence(text):
    """
    将文章转换成token + pos的dict
    """
    res = []
    p = -1
    for m in re.finditer('\n', text):
        if m.start() == p + 1:
            p = m.start()
            continue
        pos = [p + 1, m.start()]
        res.append({
            'token': text[pos[0]: pos[1]].split(' '),
            'pos': [pos[0]] + [pos[0] + loc.end() for loc in re.finditer('\s', text[pos[0]: pos[1]])] + [pos[1]]
        })
        p = m.start()
    return res


def get_train_dev():
    """
    生成train/dev的数据文件
    """
    if not os.path.exists(output_path + "srel2id.json"):
        with open(output_path + "srel2id.json", 'w') as f:
            f.write(json.dumps(rel2id))
    for dir in ['train', 'dev']:
        logger.info(f'开始预处理: {dir}')
        data = []
        for file in os.listdir(dir):
            if file.endswith('txt'):
                data += format_file(dir + '/' + file)
                # print(file)
        with open(output_path + dir + '.json', 'w') as f:
            f.write(json.dumps(data))
        with open(output_path + dir + '.txt', 'w') as f:
            for d in data:
                f.write(json.dumps(d) + '\n')


def check_label(file):
    logger.info(f'开始检查标签: {file}')
    with open(file, 'r') as f:
        data = json.load(f)
    for i in data:
        assert i['relation'] in rel2id
        assert len(i['token']) < 512
    logger.info('检查通过')


def json2txt(file):
    with open(file, 'r') as f:
        data = json.load(f)
    with open(file.replace('json', 'txt'), 'w') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')


if __name__ == '__main__':
    # 格式化train/dev
    get_train_dev()
    # 格式化test
    format_test()
    # 检查标签
    for file in ['train', 'dev', 'test']:
        check_label(output_path + f'{file}.json')