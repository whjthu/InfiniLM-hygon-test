# T09执行挂载完整SHA CPU-v2候选

root_GO/CPU_GO/GPU_GO/Model_GO=false，未远端执行。唯一未来入口为本包install_and_run.py（root注入准确INSTALL_PAYLOAD_B64）；原900秒从root安装前定epoch，不重置。仍2CPU/4GiB/0GRES，1103606/f13r3n13，唯一新CPU-v2标签。

只修v1已知prelaunch失败：当前三个版本必须精确24.05.8-3.1.3-668-20260723；完整1103606.TDB待通知占位形状不当已分配节点冲突。全部原字段/版本/分类写intent；实际--exact --gres=none --immediate=5继续由调度器准入。未知/缺字段/所选节点数字step冲突拒绝。原TDB请求及PID3940770身份仍UNKNOWN，不取消、不释放预约。

官方24.05.8 stepmgr.c 197–224/800–806/3214–3224/3429–3442支持该分类，现场厂商后缀不冒官方二进制同一。CPUs2/Nodes40不是占用测量，父预约也不是busy。

原33,306,632,192B Transformer wholeSHA、稳定FD两函数、900 epoch、cleanup/accounting、14caller链保留；仅CPU receipt标签随新版本变。before/inputs与694 UNKNOWN不触碰，CPU资格不替代完整GPU前提。实际worker后须audit_CPU_original.py一次独立终态消费；只有独审authority才可由futurecaller消费，不自动GPU。

本轮只新增classifier纯stdlib正常/恶意边界，不重旧hash/receipt/installer fixtures。verify_package.py仅校源闭包/AST，无Torch/SDK/权重读取。
