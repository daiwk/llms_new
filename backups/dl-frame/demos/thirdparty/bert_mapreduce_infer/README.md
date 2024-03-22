
# 数据格式

输入数据格式(例如,sentence1是att list，sentence2是title，其实倒过来可能更make sense)

```shell
id \t sentence1 \1 sentence2
```

# 自己训

## pretrain 

offline模式，即输入一个数据集，进行预训练

然后运行

```shell
sh -x scripts/run_train_bert_once.sh
```

大致的结果：

[https://daiwk.github.io/posts/nlp-bert-code.html#%E8%87%AA%E5%B7%B1%E5%B0%9D%E8%AF%95](https://daiwk.github.io/posts/nlp-bert-code.html#%E8%87%AA%E5%B7%B1%E5%B0%9D%E8%AF%95)

# 使用bert向量进行annoy
## 准备数据

同样的数据格式，包括miniapp和news两个文件

## 抽特征

默认使用官方的中文预训练模型

抽news

```shell
sh -x scripts/run_extract_feature_bert_hadoop.sh dict/news_title ./output/news_json_bert news
```

抽miniapp

```shell
sh -x scripts/run_extract_feature_bert_hadoop.sh dict/miniapp_title ./output/miniapp_json_bert miniapp
```

## 建annoy

因为要读所有id并建成一个annoy库，所以需要在本地跑

```shell
sh -x scripts/build_miniapp_bert_annoy.sh ./output/miniapp_json_bert ./dict/bert_miniapp.annoy ./dict/bert_miniapp_idx 768
```

## 查找相似

可以在集群跑

```shell
sh -x scripts/run_get_nearest_bert_hadoop.sh ./output/annoy_res_with_bert.all.hdfs
```

