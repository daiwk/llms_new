grep input_ids: log/run_custom_on_eval.log | awk -F'input_ids: ' '{print $2}' > input_ids
grep segment_ids: log/run_custom_on_eval.log | awk -F'segment_ids: ' '{print $2}' > segment_ids
grep input_mask: log/run_custom_on_eval.log | awk -F'input_mask: ' '{print $2}' > input_mask
