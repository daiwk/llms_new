#-*- encoding: utf-8 -*-
#!/bin/python
"""
"""
import os
import XXXX
import sys
import time
import json
import time
import requests
import shutil
import cv2


reload(sys)
sys.setdefaultencoding('utf-8')

out_dir = sys.argv[2]

def progressive_to_baseline(path, file):
    """progressive_to_baseline"""
    tmpfile = 'tmp' + file
    img = cv2.imread(path + file)

    cv2.imwrite(path + tmpfile, img)
    os.remove(path + file)
    os.rename(path + tmpfile, path + file)

def save_img(url, file_name):
    """save_img"""
    response = requests.get(url, stream=True)
    with open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    path = "/".join(file_name.split("/")[:-1]) + "/"
    fname = file_name.split("/")[-1]
    progressive_to_baseline(path, fname)

def print_res(obj, nid, idx, nid2info):
    """print_res"""
    sp_channel = int(obj.get('sparkle_channel', -1))
    content_type = int(obj.get('content_type', -1))
    vertical_type = int(obj.get('vertical_type',-1))
    sp_subcategory = obj.get('sp_subcategory',"").decode("gbk").encode("utf8")
    is_microvideo = int(obj.get('is_microvideo',-1))

    title = obj.get("title", "-1").decode("gbk").encode("utf8")
    img_url = "NULL"
    try:

        img_url = json.dumps(json.loads(obj.get("img_urls", "{}")).get("double_row_cover", "")[0]["pic_url"]).replace('"', '')
##        img_url = json.dumps(json.loads(obj.get("m_image_urls", "{}")).get("double_row_cover", "")[0]["url"]).replace('"', '')
##        double_row_cover = obj.get("double_row_cover", {})
##        print nid, 'xxxxx', double_row_cover, sp_channel, nid, vertical_type, sp_subcategory
##        if double_row_cover == {}:
##            image_cropinfo = obj.get("image_cropinfo", {})
##            if image_cropinfo != {}:
##                biserial = image_cropinfo.get("biserial", {})
##                img_url = biserial.get("pic_url", "NULL")
    except:
        image_cropinfo = obj.get("image_cropinfo", {})
        if image_cropinfo != {}:
            biserial = image_cropinfo.get("biserial", {})
            img_url = biserial.get("pic_url", "NULL")
        else:
            img_url = "NULL"

    print nid, 'xxxxx', img_url, sp_channel, nid, vertical_type, sp_subcategory
    if sp_channel == 1 and vertical_type in [11, 16] and content_type == 1 and sp_subcategory != "":
        print >> fopen, '\t'.join(map(str, [nid, "dtnews",sp_subcategory, title, img_url, nid2info[idx]]))
        if img_url != "NULL":
            save_img(img_url, "%s/%s/%s.jpg" % (out_dir, "dtnews", nid))
    if sp_channel == 1 and vertical_type in [11, 16] and content_type == 17 and sp_subcategory != "":
        print >> fopen, '\t'.join(map(str, [nid, "dtvideo",sp_subcategory, title, img_url, nid2info[idx]]))
        if img_url != "NULL":
            save_img(img_url, "%s/%s/%s.jpg" % (out_dir, "dtvideo", nid))
    if sp_channel == 1 and vertical_type == 14 and is_microvideo == 0 and sp_subcategory != "":
        print obj
        print >> fopen, '\t'.join(map(str, [nid, "video",sp_subcategory, title, img_url, nid2info[idx]]))
        if img_url != "NULL":
            save_img(img_url, "%s/%s/%s.jpg" % (out_dir, "video", nid))
    if sp_channel == 1 and vertical_type == 14 and is_microvideo == 1 and sp_subcategory != "":
        print obj
        print >> fopen, '\t'.join(map(str, [nid, "msv",sp_subcategory, title, img_url, nid2info[idx]]))
        if img_url != "NULL":
            save_img(img_url, "%s/%s/%s.jpg" % (out_dir, "msv", nid))

if __name__ == '__main__':
    BATCH_NUM = 10000
    logid = 1234
    db = XXXX.XXXXDB()
    fopen = open(sys.argv[1], 'w')
    opt = {"XXXX_srv_addr":"xxx"}
    opt['timeoutms'] = 20000
    opt["req_type"] = 1
    opt['service_name'] = 'xxxxx'
    opt["data_source"] = 3 #1.访问正排数据;2.词典数据;3.XXXX-readonly
    opt["source_type"] = 0
    db.init(opt)
    db.run()
    max_cnt = 0
    cols = ['sparkle_channel','vertical_type', 'is_microvideo', 'content_type','sp_subcategory']
    #need_lst = ["title", "url", "img_urls", "original_gimgurls", "double_row_cover", "image_cropinfo"]
    need_lst = ["title", "url", "img_urls", "image_cropinfo"]
    cols += need_lst
    nids = []
    nid2info = []
    for line in sys.stdin:
        max_cnt += 1
        parts = line.strip().split('\t')
        nid = parts[0]
        if (not nid.isdigit()) or len(nid)<12:
            continue
        nids.append(nid)
        nid2info.append(parts[1])
        if max_cnt > BATCH_NUM:
            data=db.query(logid, nids, cols)
            res_dict = dict([[k,obj] for k, obj in data.items()])
            for idx,nid in enumerate(nids):
                if nid not in res_dict:
                    continue
                obj = res_dict[nid]
                try:
                    print_res(obj, nid, idx, nid2info)
                except Exception as e:
                    print e
                    continue
            nids = []
            nid2info = []
            max_cnt = 0

    if len(nids) > 0:
        data=db.query(logid, nids, cols)
        print data, 'ooooooo'
        res_dict = dict([[k, obj] for k, obj in data.items()])
        for idx,nid in enumerate(nids):
            if nid not in res_dict:
                continue
            obj = res_dict[nid]
            try:
                print_res(obj, nid, idx, nid2info)
            except Exception as e:
                print e
                continue
    db.stop()
    fopen.close()


