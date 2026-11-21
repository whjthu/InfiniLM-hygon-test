# T11三对象：真实train2/dev1目标接线与标签绑定恢复合同

使用root已独审发布25832f8的1902B组合binding，精确SHA54ecd851…91ba8e7。训练WaterBottle/Corset各1个对象和输入，开发BoomBox1个，blind0。旧两对象“train1不足2”负证据保持原SHA；现在真实新增Corset才满足原入口>=2，不复制旧图或加入dev凑数。

`target_join3.targets_for_group(group,purpose)`只在固定新recipe下按完整binding选对象，归一化Windows/POSIX分隔符、限制项目local-only路径，再核RGBA及targetNPZ实际bytes/SHA，复用已发布原target loader。返回原`foreground_material_targets`接口的albedo[3]/两个Python scalar；未建立5字段feature cache。真实3/3 joins、dev→train拒绝，目标数据与原8RGBA/检查点审批隔离。默认沙箱小binding读取曾PermissionError；随后按root允许的精确只读auto-review授权成功，没有审批拒绝、没有改路径绕过。

## 标签身份真正参与兼容检查

`resume_labels.training_label_identity(binding)`使用按训练顺序的group、input bytes/SHA、target bundle bytes/SHA和独立recipe生成身份。dev/blind内容和物理部署路径不进入训练身份；完整来源binding SHA仍在外层证据保留。真实train标签身份为e6f401c36d52361aa776c65a498bc70a59bd1e6b45209a95d77f043afb153e78。

`domain_contract(native,binding,config,actual_native_identity=...,actual_device_ids=...)`先运行原8参数config哈希，再将它与独立train标签身份合成新的config_sha256，传给未修改的原`_sf3d_contract`。不新增原contract字段，也不把旧无label身份的hash当新hash。调用者必须提供实际native model/runtime/source身份和device IDs，本helper不生成默认模型身份。

`check_resume_metadata`执行原完整contract结构检查，并执行原`_sf3d_prepare_checkpoint`919/920两条metadata一致性检查。实际小范围CPU验证：相同label合同1PASS；旧无label绑定config、训练标签bundle变更、训练顺序变更、恢复cursor错4拒。dev label变化不影响train identity，路径迁移不影响data identity。

**上述恢复测试使用明确的合成model/runtime/device元数据身份，不是实际权重、缓存、优化器或RNG验证。** 没读任何原检查点，没有制造Tensor或feature；没有声称完整`_sf3d_prepare_checkpoint`或原Tensor检查通过。测试只用真实image/label身份隔离验证新的hash兼容路径；所有fixture身份不作为实际资格对外。

## 真正入口如何接入

在原CUDA/模型加载前完成组/文件/标签准入。模型实际加载后，原`run_sf3d_training`1088构造contract时，用此`domain_contract`替换原config_sha256构造，实际identity仍由原`_sf3d_resume_identity`获得、包含新helper源码身份。新recipe的1110标签选择调用本target join；1117–1129原生batch、scene code、query_triplane、OpenCLIP、tensor构造及loss不变。恢复时继续原完整checkpoint校验/张量/optimizer/RNG，不能仅本metadata检查就放行训练。此源码本机可调用，尚未改/部署共同backend。

原公开atlas是作者PBR UV均值recipe；随机3D query不等于UV采样点，不能把perUV标签冒3D逐点真值。仍缺实际native环境与授权模型、真实triplane/image features及其width、完整数值/真实更新恢复、新recipe参考与未打开blind对象。两train输入前提已解，但feature/model/训练/GPU/READY仍0。

新输出在 `local-only/training-sf3d-public-atlas3-resume-target-join-v1-original`，旧公开GET/数学审计不重跑、不改已发布文件、不接触集群或Git。
