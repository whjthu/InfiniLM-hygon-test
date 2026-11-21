# 固定 CMU→WoodenMesh 颈部枢轴审计

只消费已冻结录制与骨架/统计，不下载、不改旧包、不生成正式缓存。`audit_pair.py` 完成一次470帧可逆配对；`rest_evidence.py` 只查静态骨段跨度，不是第二候选。所有数组留local-only。原128/342是同录制前段/后段，不是独立留出或泛化证据。

CMU真实FK940帧均精确满足 lowerneck.tip=upperneck.base、upperneck.tip=head.base。现Neck←lowerneck.tip与Head←upperneck.tip的出射段是upperneck，故唯一候选用原upperneck世界旋转（不是lowerneck）确定twist。先使用原rest基与相同30Hz滤波，再最短swing投影到原Neck→Head方向，不用统计选择角度。全部非Neck195字段含root与66关键点逐字节保留；独立FK全52节点位置变化仅3.623e-9 rig units。其它关节的本地旋转字段不变；作为终端子节点的Head世界朝向可随颈部twist改变，未冒其世界朝向不变或真实测量已知。

实际拒绝：后342 Neck RMS z 9.502718→9.785606，最大|z|24.755720→24.768044；全470 RMS9.631600→9.844021。无裁剪、夹位、换均值、重选帧或参数网格。配对文件含前后原值，另一次存盘被动检查验证恢复原Neck6列后全201位一致。

几何不可达证明：固定a=Wood Neck→Head rest单位轴，b=原本地Neck旋转后的同轴。对任意保持当前位置的旋转R且R a=b，R11的精确区间为a_y b_y±sqrt(1−a_y²)sqrt(1−b_y²)。真实470帧的区间上界全部小于0；原公开统计Mean[78]=0.94839379、Std[78]=0.06377848。所有470帧即使任意twist仍|z|>10，最大不可消除下界26.090818，几乎就是原26.091595。因此继续扫颈轴offset不能解决当前大swing。

静态incoming跨度也有真实差异：原CMU thorax.tip→lowerneck.tip经已冻结腿长尺度为0.0931705，Wood Spine3→Neck为0.1968452，比例0.4733；而当前Neck→Head outgoing比例0.9737。这说明当前跨rig锚点/身体比例需要重新验证，不能仅靠名称认定同一解剖枢轴。当前结果不证明某个新重锚方案正确，也不允许悄改既有位置或其它关节来压z。本轮无第二候选，source135/201与正式训练资格仍未建立，READY+0。

源身份见SOURCE_PROOF及SOURCE_FILES；原caef/3fba及6°负结果完整保留。下一必要工作是独立枢轴/几何约束证据，而非再扫twist或将原统计当目标动作。
