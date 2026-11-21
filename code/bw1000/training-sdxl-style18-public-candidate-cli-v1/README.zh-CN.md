# SDXL Style18 独立公开候选：共同 backend 的最小接口修复

只复用缓存，不下载。原Apache-2.0官方commit `a3608b512ed7248499a44c61d954965ed9bdae4d` 56280B源码加原冻结17 hunk后的候选为63989B/f3f938…；它不是原A100 15bf8b… trainer，旧T06数值FAIL、恢复源码不匹配及来源失败历史均保留。本包明确是一份新领域配方的候选源码适配，不改原17或原trainer。

实际查出两个阻塞：196501…共同CUDA/HIP backend发出4个`--gencuda_*`参数，但候选只有`--genmlu_*`；候选唯一显存观测调用还是`torch.mlu.max_memory_reserved()`。`prepare_candidate.py`只添加4个CLI别名（首名称仍genmlu，dest和内部代码不变），将该一处观测改成`torch.cuda.max_memory_reserved()`。精确反补丁后逐bytes等于原f3f938，Apache头完整保留。torch.cuda在目标HIP的实际可用性仍需原运行时资格，未在本机调用该API。

生成的64125B源码SHA为 `b3e85543eaeeb42edd67dca0ffbc54391e4d8fa33b44aa1522e24b31ffc6c7bd`，只在 ignored `local-only/training-sdxl-style18-public-candidate-cli-v1-original/candidate-source-root/examples/text_to_image/train_text_to_image_lora_sdxl.py`。不把官方正文复制进Git。生成器以精确f3f938缓存为输入，不能接受另一版本/重下载。根部署时应保持这一文件的新领域修改说明及原Apache署名/许可证，并另锁新source_root和发布身份；不能覆盖原A100或旧source_root。

本机实测8项源接口通过：实际196501 backend命令AST切片生成fresh argv，原候选解析器拒绝，新解析器接受全部参数；256/rank4/alpha由原rank派生4/batch1/accum1/BF16/constantLR/梯度检查点off/文本编码器不训练均保持。解析器只执行parse_args函数，动态插值枚举的名称用明确fixture供应，不是torchvision资格。没有import trainer/Torch、ImageFolder、tokenizer、VAE或模型。另实际调用现有`preflight_sdxl_style18_target.inspect_source`，对本机新源和精确196501 backend核20必需flags通过；结果虽沿原函数命名为DEPLOYED_SOURCE_VERIFIED_NATIVE_IO_PENDING，本轮scope严格只是本机私有源码，**不是目标部署PASS**。

新鲜25步之外仍有硬阻塞：

- 原backend的`warmup_steps=5`只排除前5步计时；实际argv是constantLR与lr_warmup_steps0，不能声称有5步学习率预热。
- backend外部恢复只发`--gencuda_external_resume_from_checkpoint`，候选main实际要求full-checkpoint显式开启；现配方workers2，full-checkpoint要求workers0。测试实际执行了候选main的第一条守卫，确认拒绝。不能称本包已支持25+5，不能只补一个恢复参数冒RNG/游标准确。
- 新source要求真实完整SDXL pipeline根：两个tokenizer、两个text encoder、scheduler、VAE、UNet，以及最终pipeline读取的model_index.json。原A100依赖只提供UNet 10270077736B/357650…等身份，不能据此宣称目标七组件完整。本轮没有获得/假造新的目标完整模型manifest或当前权重路径，根须用现存已授权资产证明，无新权重传输。
- 原生输入需真实ImageFolder `metadata.jsonl`的`image`/`text`、两tokenizer、resize+centercrop256、[-1,1]tensor、两编码器及VAE/UNet前向实际通过。当前12train/6dev仅数据资格；无dev/盲图或标签读取混入本包。caption是已有style说明，图表数字/问答金标未用。ImageFolder递归路径只能给train子目录。

根的下一最短动作：把此新候选单文件部署到独立source_root，按原source SHA核本地/目标一致，再运行已有预检（其不加载模型）：

```
python -B tools/preflight_sdxl_style18_target.py --train-dir /public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-sdxl-USGS-style18-domain25-data-v1/train --source-root NEW_DOMAIN_SOURCE_ROOT --backend VERIFIED_196501_BACKEND_PATH --expected-hygon-trainer-sha256 b3e85543eaeeb42edd67dca0ffbc54391e4d8fa33b44aa1522e24b31ffc6c7bd
```

随后只有完整权重/环境、同输入数值及唯一设备准入闭合，才可用原portable `entry.run_via_common_backend(backend, weights=REAL_SDXL_ROOT, output_dir=NEW_RUN, source_root=NEW_DOMAIN_SOURCE_ROOT, train_data_dir=EXACT_TRAIN_DIR, trainer_sha256=上述新SHA)`。该原入口本就接受独立部署trainer SHA，无需新训练loop。当前CPU/GPU/训练/Model/READY均false，根负责派发和发布。

官方来源：[固定脚本](https://github.com/huggingface/diffusers/blob/a3608b512ed7248499a44c61d954965ed9bdae4d/examples/text_to_image/train_text_to_image_lora_sdxl.py)、[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)、[USGS版权与署名](https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits)。本轮未重新联网；来源/数据分组沿已发布冻结材料。
