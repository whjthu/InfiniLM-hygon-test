# O01 48 GiB 见证后的普通八卡恢复

此入口只接收当前 48 GiB 已发布 stage、根独占 pending marker 和真实独立清场之后的新 refill grant。它按封存 SHA 复用曾实际恢复 .2179 的三份既有本机私有源码，保留普通 controller、独立设备准入、单次发送记录和双点复核。只替换旧 stage、当前 pending 文件和双保护 bridge 的固定引用。源码和认证配置不随 Git 发出。需要在原有已授权通道工作区运行。

    python3 -I -S -B refill.py launch --stage LOCAL_STAGE --receipts PRIVATE_DIRECTORY --request GRANTED_ID --publication APPROVED_COMMIT --marker-sha SHA --grant-sha SHA
    python3 -I -S -B refill.py consume --stage LOCAL_STAGE --receipts PRIVATE_DIRECTORY --request GRANTED_ID --publication APPROVED_COMMIT --marker-sha SHA --grant-sha SHA

根先通过独立会计确认 48 GiB GPU 与零卡内存观察者实际终结，再让已发布 HOST post 取得同 boot、完整八 PCI、设备清洁证明及唯一 grant。已明确未提交 GPU 的路径还须见到原 sidecar 真实编号终态。授权发送只在同一有限连接获得 fresh 物理节点证明后构造，之后保留一次性本机 send token。controller 的双保护过滤器核同一 grant 和同一节点。目标返回未知时保留原 request、scope 和 token，不重放。

只有独立零卡 observer 与普通八卡 worker 账实匹配，同一 PID/startticks 的八 PCI 两点 GEMM 都增长后才消费并解除保护。此前仍排除其他补位入口。未部署或 BOUND 仅为工程资格。未执行新的现场动作。
