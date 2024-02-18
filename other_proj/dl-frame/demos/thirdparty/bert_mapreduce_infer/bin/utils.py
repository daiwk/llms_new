#!/usr/bin/env python
# encoding: utf-8
"""
utils
"""

import numpy as np
import globals

def load_whitelist_tag():
    """
    load_whitelist_tag
    """
    whitelist_tag_set = set()

    with open(globals.whitelist_tag_file, "rb") as fin_whitelist_tag:
        for line in fin_whitelist_tag:
            whitelist_tag_set.add(line.strip("\n"))
    return whitelist_tag_set


def load_blacklist_tag():
    """
    load_blacklist_tag
    """
    blacklist_tag_set = set()

    with open(globals.blacklist_tag_file, "rb") as fin_blacklist_tag:
        for line in fin_blacklist_tag:
            blacklist_tag_set.add(line.strip("\n"))
    return blacklist_tag_set


def load_bert_idx2nid_dict_from_miniapp_annoy():
    """
    load_bert_idx2nid_dict_from_miniapp_annoy
    """
    ## load idx->miniapp_nid
    with open(globals.bert_idx2nid_file, "rb") as fin_idx2nid:
        dic_idx2nid = {}
        for line in fin_idx2nid:
            line = line.strip("\n").split('\t')
            idx = int(line[0])
            nid = line[1]
            dic_idx2nid[idx] = nid
        return dic_idx2nid


def func_load_file_index():
    """
    func_load_file_index
    """
    file_index_list = []
    with open(globals.file_index_file, 'rb') as fin_file_index_file:
        for line in fin_file_index_file:
            file_index = line.strip("\n")
            pair = (None, {"file_index": file_index})
            file_index_list.append(pair)
    return file_index_list


def func_load_file_index_mutable():
    """
    func_load_file_index_mutable
    """
    file_index_list = []
    with open(globals.file_index_file, 'rb') as fin_file_index_file:
        for line in fin_file_index_file:
            file_index = line.strip("\n")
            pair = [None, {"file_index": file_index}]
            file_index_list.append(pair)
    return file_index_list


def func_load_file_index_bert():
    """
    func_load_file_index_bert
    """
    file_index_list = []
    with open(globals.file_index_bert_file, 'rb') as fin_file_index_file:
        for line in fin_file_index_file:
            file_index = line.strip("\n")
            pair = (None, {"file_index": file_index})
            file_index_list.append(pair)
    return file_index_list


