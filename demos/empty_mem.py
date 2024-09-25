def xx():            
    batch_num = _gid_embs_cpu.size(0) // batch_size + int(
            _gid_embs_cpu.size(0) % batch_size != 0
    )
    last_log_time = -1
    for i in range(batch_num):
        start, end = i * batch_size, (i + 1) * batch_size
        with torch.no_grad(): ## code by O1-preview..
            # Move the slice of CPU tensor to GPU
            tmp_tensor = _gid_embs_cpu[start:end].to(_device)
            
            # Perform the operation on the GPU tensor
            lm_tmp_tensor = lm_tower(tmp_tensor)
            
            # Copy the sum into the target GPU tensor using in-place operations
            _gid_embs[start:end].copy_(tmp_tensor).add_(lm_tmp_tensor)
            
            if _l2_norm_type in ["g", "g_u"]:
                tmp_tensor_norm = _l2_norm(_gid_embs[start:end])
                _gid_embs[start:end].copy_(tmp_tensor_norm)
                del tmp_tensor_norm
                # logging.info("use l2 norm")
            # Delete temporary tensors to free up memory
            del tmp_tensor, lm_tmp_tensor
            
            # Optional: Clear CUDA cache to free up fragmented memory
            torch.cuda.empty_cache()
