#encoding=utf8

#### ************************************
###### news related files
tuwen_bert_file = "./output/news_json_bert"

#### ************************************
blacklist_tag_file = './dict/blacklist_tags'
stopwords_file = "./dict/stopwords"

#### ************************************
###### bert annoy related files 
bert_idx2nid_file = "./dict/bert_miniapp_idx"
bert_miniapp_annoy_file = "./dict/bert_miniapp.annoy"

#### ************************************
###### threshold settings

g_upper_threshold = 0.35
g_lower_threshold = 0.35


g_max_print_cnt = 10 # save 10res for ranking

#### ************************************
###### productioin output files

file_index_bert_file = "./dict/file_indices.tuwen.bert"

annoy_res_with_bert = "output/annoy_res_with_bert"

