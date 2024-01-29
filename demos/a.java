// debug logs
LOGGER.info("dwk test {} g_ecom_live_product_good_new_ctg1: {}", row_key, candidate_info.getOrDefault("g_ecom_live_product_good_new_ctg1", 0));
LOGGER.info("dwk test {} g_ecom_live_product_good_new_ctg2: {}", row_key, candidate_info.getOrDefault("g_ecom_live_product_good_new_ctg2", 0));
LOGGER.info("dwk test {} g_ecom_live_product_good_new_ctg3: {}", row_key, candidate_info.getOrDefault("g_ecom_live_product_good_new_ctg3", 0));
ArrayList<Long> tmp_lst1 = ((ArrayList<Long>) candidate_info.getOrDefault("fc_g_ecom_live_product_goods_pred_ctg1_list_full", new ArrayList<Long>()));
String result1 = tmp_lst1.stream()
                .map(String::valueOf)
                .collect(Collectors.joining(";"));
LOGGER.info("dwk test {} fc_g_ecom_live_product_goods_pred_ctg1_list_full: {}", row_key, result1);
ArrayList<Long> tmp_lst2 = ((ArrayList<Long>) candidate_info.getOrDefault("fc_g_ecom_live_product_goods_pred_leaf_ctg_list_full", new ArrayList<Long>()));
String result2 = tmp_lst2.stream()
                .map(String::valueOf)
                .collect(Collectors.joining(";"));
LOGGER.info("dwk test {} fc_g_ecom_live_product_goods_pred_leaf_ctg_list_full: {}", row_key, result2);
