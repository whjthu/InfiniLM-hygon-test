# 同节点正常8DCU分配下的只读元数据对照

全GO=false，未远端/提交。复用353成功诊断的stage_worker.py exactbytes，不重原0GRES或80节点采样。唯一声明资源变量为--gres=none→--gres=dcu:8；仍1087656/a01r4n14/1CPU256MiB，worker20/held25+5/实际Elapsed<=30/Slurm最小60/总epoch90不增。8*60=480DCU秒(0.133333卡时)只是调度资源预留上限，0GPU计算/SDK/busy计数，不算训练利用率。

入口必须gitshow原80失败及353基线两个公版SHA。actualAllocTRES严核8DCU、1CPU、1node、256MiB且无其他GRES；ReqMem空不误拒。与353保存的node/UID/boot逐项相同，否则留新结果但对照不资格化。所有原row/metadata失败/cleanup保留。元数据观测本身捕获EPERM时worker仍正常C0，metadata_inventory_passed另列，不冒采样PASS。

root审后只翻root_GO/CPU_GO/GPU_GO/GPU_resource_GO四位并重MF/pub；GPU_compute_GO/Model_GO/sampler_GO必须false。run_once.py --publication <40hex> --request-id <fresh32hex> 是未来唯一入口；固定rootPython/cwdG/既有ordinarySSH。无sudo/ACL改动/新服务、无GPU初始化、无待许可Torch事实。若8GRES下成功仅支持资源分配相关的访问差异，不能冒已证明具体cgroup机制；在结果前不另full640采样。
