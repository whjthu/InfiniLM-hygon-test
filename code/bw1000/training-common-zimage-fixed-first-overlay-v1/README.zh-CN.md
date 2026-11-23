# T07.sft 固定首步入口

统一入口在已验证的统一训练候选中接受下面的诊断命令。

```bash
python3 -B tools/hygon_training_recipe.py T07.sft --fixed-first-update \
  --binding /absolute/native-binding.json \
  --reference /absolute/original-first-update-private \
  --reference-sha256 "$REFERENCE_PROGRESS_SHA256" \
  --output /absolute/new-output-directory \
  --epoch "$RUN_EPOCH" --deadline "$RUN_DEADLINE"
```

调用方用已发布的任务范围指定执行平台、原配方和输入身份。参考目录保留原 `progress.json`、`prepared.pt` 和 `target.pt`。入口核对清单和每个文件的原始字节，执行零更新前向及一次正式配方下的原生反向和 AdamW 更新。零更新前向之后恢复 RNG 并检查参数、梯度、缓冲区、调度器和优化器边界。单次诊断的张量和私有模型输出留在实验端。

海光真实准备张量的 CPU 验证见 [原步骤 1103606.1693](../../../results/bw1000/2026-11-23/training-zimage-recorded-reference-original1693-accounting.json)。跨平台验收需要完整原模型实跑、原 A100 同输入及原阈值。独立恢复、正式吞吐及留出集各自保留验收证据。
