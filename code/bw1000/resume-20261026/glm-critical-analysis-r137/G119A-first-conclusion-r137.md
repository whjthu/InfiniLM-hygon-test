GLM767 G119A — 真实结果首判

767 terminal COMPLETED0，all16审计和post KFD均通过。M15 OFF，row2 ON；同32arrival，一次加载。

|episode|task tok/s|decode CAPI tok/s|visible|QA/EOS/cap/unfinished|
|---|---:|---:|---:|---|
|controls-before|13.170224|22.716397|2215|31/32, 32, 0, 0|
|sparse-profile|13.243602|22.710468|2215|31/32, 32, 0, 0|
|controls-after|13.197909|22.606088|2215|31/32, 32, 0, 0|

task control drift 0.2100%, profile effect 0.4516%; decode drift 0.4868%, effect 0.2172%。均低于预设3%。每owner自然T16×3、T15×3、B8/B1 PREFIX各1；max144 events<512，完整sequence/row,past,count审计通过。

|stage1/rank0实际selector(seq)|CAPI ms|incoming GPUstream ms|local layers GPUstream ms|full attn/FFN ms|shared attn/FFN ms|
|---|---:|---:|---:|---:|---:|
|B8 PREFIX(1)|12808.902|6153.376|6652.991|62.381/107.815|56.167/108.181|
|T16(5)|525.240|251.048|272.086|7.318/4.764|2.850/3.809|
|T16(6)|512.713|244.474|266.035|3.485/3.958|2.927/3.831|
|T16(7)|510.698|244.538|264.098|3.491/3.890|2.813/3.876|
|B1 PREFIX(26)|2683.812|1287.582|1393.794|14.486/22.925|11.530/22.793|
|T15(124)|864.400|419.346|441.720|6.696/4.530|6.540/4.518|
|T15(125)|865.321|419.315|442.750|6.717/4.531|6.552/4.491|
|T15(126)|874.015|419.294|451.215|6.788/4.572|6.590/4.725|

6个pipeline值分别为HOST incoming/layers/outgoing、GPUstream incoming/layers/outgoing；HOST分桶会因等待落点改变，不能把HOST层包络当纯CPU或GPU计算。profile PREFIX pooled CAPI66.995s：HOST incoming0.00153/layers66.979/outgoing0.00968s，GPUstream incoming32.127/layers34.853/outgoing0.00915s。DECODE CAPI96.123s：HOST0.01779/95.792/0.11623s，GPUstream46.252/49.557/0.10978s。HOST与GPU是嵌套同一工作，两PP stage也不得相加。

首个B8 PREFIX：layer38 attention HOST6.215531s，GPUstream仅0.062381s；前面incoming GPUstream6.153376s。T16 seq6对应HOST attention247.772ms/GPU3.485ms，incoming244.474ms。同一依赖在首层HOST attention中阻塞，不能误报attention慢或可删除CCL。

FFN在B8/B1两个代表层确实分别约108/23ms，大于该层attention约56–62/12–14ms；T16两个代表层FFN约3.8–4.0ms、attention约2.8–3.5ms。仅测2层detail，不乘40/78层，不称专家失衡；缺少真实专家/job计数及内部CCL/readiness。

当前 G119B PARK，源码patch为0。最大已测整调用区间是上游生产依赖，尚未证实可删；现有本地完整marks只含emitter，其他15 owner仅资格摘要。root可单次读取已经保留的16owner数据，不新增profile或加载。只有该现成数据支持一个>=约10%实际criticalwall可改区间时才准备一个patch；否则停止本路线。
