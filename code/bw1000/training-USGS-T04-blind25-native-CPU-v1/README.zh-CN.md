# USGS 固定盲组25题原生输入 CPU 资格

独立新 scope，原 blind 5 图 / 3 事件闭包组 / 16 源事件 / 25 QA，不改名 dev。这里只处理原 user+image，不取答案，不读训练/开发集，不加载权重、不初始化 CUDA、无更新。原 native55 仅 train45/dev10，本包不重跑55或继承其盲组资格。

普通发布和 root 采用后，唯一入口 `submit_once.py --publication <actual_pub> --root-go`。首行捕获单300秒 epoch，包含本机核源/Git/私有输入图读取、上传、目标安装和CPU；native260、stop270、held282、tree285、alarm290、client295。2CPU排除0/1，RSS-v2按同owner startticks前后核采样，8GiB为 sampled soft stop，不冒硬内存限制或缺失RSS为0。清理仅本进程真实持有Popen及其owned后代。未知执行不得重播。

原统一396 `tools.hygon_vlm_recipe.build_launch`+原T04 HF插件，原14模型元数据精确复制至metadata-only视图，7Swift与3额外wrapper源逐SHA核。load_model=False，transformers mode，完整4096上限、max_pixels262144、truncation=raise、collator标签None；无新提示词/FINAL/EOS字面量。原504 LoRA绑定仅静态来源，不进行初始化资格。原生25题全部保留，失败记录完成分母和partial identities。

`read_terminal.py --scope <namespace> --request-id <actual> --publication <pub> --epoch <origin> --output <new_private_json>` 仅在原controller实际终结后运行一次。只审元数据/25字段身份与source-ID哈希、wrapper链、RSS和独立post，不再processor编码，不读题面/图/gold。原producer失败原样保留。

本包GO初始全部false。原三检查点与25更新、greedy/cap256/评分函数已在dev结果后固定；本次不根据盲组换检查点/训练/调参。CPU通过后另scope产生3×25=75完整预测，全部持久封存后才能解封gold并计全部missing/cap/wrong。此小型地震报告读图集不代表正式17用例或领域优势；dev10 domain=generic均9/10，尚无领域优势。
