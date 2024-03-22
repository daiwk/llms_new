import sys
in_file = sys.argv[1]

dic_full2id1 = {}
dic_full2id2 = {}
dic_full2id3 = {}
with open(in_file, "r") as fin, \
    open("first.txt", "w") as fout_1, \
    open("first.txt_mapping", "w") as fout_1_m, \
    open("second.txt", "w") as fout_2, \
    open("second.txt_mapping", "w") as fout_2_m, \
    open("third.txt", "w") as fout_3, \
    open("third.txt_mapping", "w") as fout_3_m:
    for line in fin:
        line = line.strip("\n").split("\1")
        cate1_id, cate1_name, cate2_id, cate2_name, cate3_id, cate3_name, cnt = line
        cnt = int(cnt)
        full_cate1_name = cate1_name
        full_cate2_name = cate1_name + "##" + cate2_name
        full_cate3_name = cate1_name + "##" + cate2_name + "##" + cate3_name

        dic_full2id1[full_cate1_name] = cate1_id
        dic_full2id2[full_cate2_name] = cate2_id
        dic_full2id3[full_cate3_name] = cate3_id

    for cate1_name, cate1_id in dic_full2id1.items():
        fout_1.write("{}\n".format(cate1_name))
        fout_1_m.write("{}\t{}\n".format(cate1_name, cate1_id))
    for cate2_name, cate2_id in dic_full2id2.items():
        fout_2.write("{}\n".format(cate2_name))
        fout_2_m.write("{}\t{}\n".format(cate2_name, cate2_id))
    for cate3_name, cate3_id in dic_full2id3.items():
        fout_3.write("{}\n".format(cate3_name))
        fout_3_m.write("{}\t{}\n".format(cate3_name, cate3_id))

print("done")

