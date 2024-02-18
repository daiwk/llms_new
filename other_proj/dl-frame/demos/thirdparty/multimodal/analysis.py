#encoding=utf8
import sys

f_recall_eyes = sys.argv[1]
f_recall_mouth = sys.argv[2]
f_tp_eyes = "all_tp_eye.nids"
f_tp_mouth = "all_tp_mouth.nids"



with open(f_recall_eyes, "rb") as fin_recall_eyes, \
    open(f_recall_mouth, "rb") as fin_recall_mouth, \
    open(f_tp_eyes, "rb") as fin_tp_eyes, \
    open(f_tp_mouth, "rb") as fin_tp_mouth:

    recall_eyes_set = set()
    for line in fin_recall_eyes:
        line = line.strip("\n")
        recall_eyes_set.add(line)

    recall_mouth_set = set()
    for line in fin_recall_mouth:
        line = line.strip("\n")
        recall_mouth_set.add(line)

    tp_mouth_set = set()
    for line in fin_tp_mouth:
        line = line.strip("\n")
        tp_mouth_set.add(line)

    tp_eye_set = set()
    for line in fin_tp_eyes:
        line = line.strip("\n")
        tp_eye_set.add(line)


    title = ["模型", "交集", "召回结果", "真实结果", "交集/真实"]
    print "\t".join(map(str, title))
    inter_eyes = recall_eyes_set & tp_eye_set
    lst = [f_recall_eyes, len(inter_eyes), len(recall_eyes_set), len(tp_eye_set), float(len(inter_eyes)) / len(tp_eye_set)]
    print "\t".join(map(str, lst))
    inter_mouth = recall_mouth_set & tp_mouth_set
    lst = [f_recall_mouth, len(inter_mouth), len(recall_mouth_set), len(tp_mouth_set), float(len(inter_mouth)) /  len(tp_mouth_set)]
    print "\t".join(map(str, lst))
