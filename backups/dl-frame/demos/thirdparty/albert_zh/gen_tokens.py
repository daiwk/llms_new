import sys
import tokenization

tk = tokenization.FullTokenizer("./albert_config/vocab.txt", False)

file = "./my_set/test.txt"
file_token = "./input_ids.gen"
file_mask = "./input_mask.gen"
file_segment = "./segment_ids.gen"
idx = 0
max_len = 128
max_save = 218
with open(file, 'r', encoding="utf8") as fin, \
    open(file_token, 'w', encoding="utf8") as fout_tokens, \
    open(file_mask, 'w', encoding="utf8") as fout_mask, \
    open(file_segment, 'w', encoding="utf8") as fout_segment:
    for line in fin:
        if idx == 0:
            idx += 1
            continue
        if idx > max_save:
            break
        line = line.strip("\n").split("\t")
        s1 = line[0]
        s2 = line[1]
        tokens = ["[CLS]"]
        s1_tokens = tk.tokenize(s1)
        tokens += s1_tokens
        tokens.append("[SEP]")
        s2_tokens = tk.tokenize(s2)
        start_s2 = len(tokens)
        len_s2 = len(s2_tokens)
        tokens += s2_tokens
        tokens.append("[SEP]")
        id_tokens = tk.convert_tokens_to_ids(tokens)
        real_len = len(id_tokens)
        id_tokens += [0] * (max_len - real_len)
        str_token = " ".join(map(str, id_tokens)) + "\n"
        fout_tokens.write(str_token)
        id_mask = [0] * max_len
        for x in range(0, real_len):
            id_mask[x] = 1
        str_mask = " ".join(map(str, id_mask)) + "\n"
        fout_mask.write(str_mask)
        print(start_s2, len_s2, 'ooo')
        segment = [0] * max_len
        for x in range(start_s2, start_s2 + len_s2 + 1):
            segment[x] = 1
        str_segment = " ".join(map(str, segment)) + "\n"
        fout_segment.write(str_segment)
        idx += 1

