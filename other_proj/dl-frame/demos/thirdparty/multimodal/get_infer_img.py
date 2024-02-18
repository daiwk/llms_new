import sys
import requests
import shutil


file_in = "badcase.txt"

def save_img(url, file_name):
    """save_img"""
    response = requests.get(url, stream=True)
    with open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)

with open(file_in, 'rb') as fin:
    for line in fin:
        line = line.strip("\n").split("\t")
        nid = line[0]
        img_url = line[3]
        save_img(img_url, "imgs_infer/%s/%s.jpg" % ("dtnews", nid))
