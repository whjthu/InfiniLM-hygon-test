# W2 INT8 v2：寄存器scale/累加（仅CPU源码）

v1已PARK：T8四对全负、合成低复用T64四对全负；高复用历史T64正例不推Model。v1源码和结果均未覆盖。

v2只替换matrix函数。使用750/751/753已验证的DU accumulator位置：每lane固定row=lane&15，四个col=(lane>>4)+4*i。在同一寄存器持有对应四个FP32 sum；六个独立I32 group32各自scale后按g升序累加。rid<0只跳过scalar scale，MMAC必须仍由全64lane执行，因为这些lane的B碎片也用于有效A行。六组结束后一次FP32 rowmajor shared transpose，再用原coalesced写partial。quant/jobs/W13/reduce、activation量化、全局scratch及三个scenario不变。

sourceproof枚举每输出的旧/新owner和g链，证明256输出双射、有效行写读集合一致、无无效tmp读取；矩阵函数外原始字节及runner/consumer/header原字节一致。静态tmp流量从每job×Ntile 12288B变为128×validM B（最多2048B），scalar scale从1536个元素变为96×validM，source barrier20→15；这些是代码数据流计数，不能据此声称动态瓶颈或速度收益。LDS仍2112B，VGPR/spill变化未知。

等待root将新机制认领普通push后才编译。最小编译：使用实际750 nvcc/flags及同TU -S，只替换provider.o；其余旧对象和MD逐项SHA核验。旧751 runner DSO绝对DT_NEEDED绑定v1 native，因此需要从原字节不变的wrapper源码按原命令链接新private native，ABI不变，不创造新HOST流程。检查真实6条W2 I8及原warm原语、无F64、关键header/ldd-r闭包，另必须核实编译器没有把MMAC下沉到validrow exec mask中、没有每组I32 tmp往返。若无法确认或出现spill/闭包扩展，STOP。

当前未SSH、未编译、未Slurm、未GPU、未读权重，也没有新CPU性能/独立门。后续GPU仍需源码发布及root单独GO。
