# MiniCPM USGS88：真实训练前的原生输入资格

只使用已在目标端的原 USGS88 训练记录和22张派生表格，精确dataset SHA与Qwen493相同。没有新GET、传输、图片生成或开发/盲测读取；同数据不代表MiniCPM已经资格通过。这里预注册176次原生encode（完整模板与4096模板各88）及88次真实native collator，失败保留88样本及全部调用分母。它是训练必要前提，不计真实optimizer更新或training READY。

原MiniCPM完整10模型已有独立证据，旧传输task草稿的INCOMPLETE不作为重传理由。CPU只复制9个小metadata到新物理 `MiniCPM-V-4.6` 目录（20035101B），无 `model.safetensors`。load_model=False、offline、隐藏三个GPU变量；实际模型应返回None，未调用模型forward或optimizer。原冻结初值目标文件43466515B/d3824dc5…只在HOST校验bytesSHA，不解序列化；原named9ac0f645…及完整模型的历史证据保留，未来训练仍须真实恢复该原件，不用seed替代。

调用构造沿用真实CPU119：`get_model_processor(...,model_type="minicpmv4_6",load_model=False,download_model=False,device_map="cpu",attn_impl="sdpa",torch_dtype=BF16)`，`get_template(...template_type="minicpmv4_6"...)`，train模式。新增完整/4096模板及raise策略核未截断；EOS按MiniCPM实际ChatML suffix `<|im_end|>\n` 的原生 `_encode_context_list` 编码验证，完整样本尾部须精确相同且其中EOS受监督，允许真实换行但不允许任意后续token，另列2次suffix元数据编码；原.119没有collator实测。既有Swift4.4.2原base源码给出的公共签名为 `data_collator(self,batch,*,padding_to=None)`；本次仍在任何encode前对实际绑定方法做signature.bind，并逐样本真实调用验证，不能把基类API或Qwen资格当MiniCPM已通过。

实际get_model_processor/get_template及encode/collator/set_mode所有wrapper层分别记录，Swift/HF源须逐bytes等于已经缓存且整wheelSHA通过的官方ms-swift4.4.2和Transformers5.12.1。Torch context wrapper单列DTK来源和实际源SHA，不冒充Swift实现或新官方wheel证明。记录实际字段hash、长度、EOS、标签mask、监督token、CPU有限值、RSS和进度；只持久化身份和计数，原token/文本不输出到PC。调用者不读取模型权重，隐式native-loader的GPU行为仍标UNKNOWN；只检查自己进程前后设备FD，不声明全节点GPU独占。

候选CPU资源1103606/f13r1n19，8CPU/12GiB/0GRES，whole300、Slurm≤240、worker≤210，均同一原epoch缩水，含本地Git、传输与资产检查。沿已成功绝对Apptainer、CPU无--rocm、合格hyhal系统alias只读bind；现有runtime和wheel只读，不安装包。原T05未来训练仍保持5步预热、GC OFF、SDPA、冻结视觉与merger、LoRA16/32/.05、原AdamW/cosine，不改正式17配方。

`launch.main(publication,transport)` 仅由root现有普通SSH回调前台调用；独占新scope和固定request，UNKNOWN不重播。`read_terminal.py` 已预置：只读原source/intent/实际accounting/原生字段metadata，不重processor/数据/图像/权重，不查询新作业。CPU实际通过并独审后才准备新的原25步训练候选。原151恢复数值漂移与未知内部诊断均不解决、不重放；不能借本次CPU输入资格把它们洗PASS。

当前只批准一次 CPU 输入验证，GPU、训练和模型仍待验收。只完成source、Python3.10语法及新增合成mask/EOS边界检查；实际176encode、88collator、模型初值值核验、GPU训练与效果均未执行。
