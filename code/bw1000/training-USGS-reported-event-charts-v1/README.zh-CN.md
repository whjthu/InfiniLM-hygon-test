# USGS reported-event chart reading：离线小型数据 scope，未执行

原查询已保存35,802B并独审50/50数值槽、50alias组、12源地理组。原producer count KeyError、C1与source_accessible=false永久保留。本包仅依赖独审的精确source-records/result SHA，不再GET、下载或安装，不重跑旧源/renderer资格。50是实际源分母，原limit50不意味着全周完整目录。

在任何QA/gold之前，按共享alias、同原30°纬/60°经cell、或时间差≤72h且球面距离≤500km连边，取传递闭包。全部事件/图窗/视觉变体/派生题共组；SHA(seed+sorted-member-groupSHA)排序按约60/20/20组切train/dev/blind。地区/邻近簇不能跨split，实际组数<3即拒绝，不换事件或放宽距离。该保守几何切分仅小pilot，不能冒独立场景/地震学泛化。

每组按原time/id顺序固定≤5事件图窗，覆盖全部50槽恰一次，最多36张512×512 PNG/180题、120s/2CPU2GiB；实际数量由固定分组决定，不凑上限。每窗5题：E01报告震级数字、E01深度km、末行减首行显示深度、最大深度全ties、最大报告震级数字全ties。单事件窗保留且显式标记，差值0不是独立复杂推理。所有源Decimal先ROUND_HALF_UP两位显示；gold只从显示值重算，负零统一0.00。不同magType只读报告数字，不比物理能量/风险。图含UTC、km、magType、显示精度、USGS署名及本地派生声明。字体/Matplotlib版本/dpi/pixels/Software metadata固定；PNG身份全SHA。小字体可读性待真实train/dev视觉复核，不冒已合格。

输出忽略local-only，split-plan先持久化；inputs.jsonl只有user/messages/images，labels-private独立、dev/blind答案绝不进入训练接口。未来同一统一框架五VLM按各原生模板/collator接该独立domain recipe，不新训练loop，不改原17/8卡BF16参考口径。任何卡数/计时/有效监督token变更须新参考。score_once只支持train/dev，按显式natural_EOS=true、cap_hit=false计正确，缺/错/截断/未EOS全分母；task exact accuracy及chart/group统计须与base配对，无loss冒效果。

真实构建后，视觉审查只能train/dev：每split按chart id SHA排序，固定最早1单事件窗(若有)+最早1多事件窗(若有)，每split最多2图，无替补挑好看图、不读blind图/gold。审源显示值/单位/labels/table及图表可读性，失败保失败。后续独立reader核所有图身份/分组/QA可重算，但不得人工盲标签调参。图/QA/训练/领域模型产出目前均未发生，GO全false。

版权依据：[USGS官方FAQ](https://www.usgs.gov/faqs/are-usgs-reportspublications-copyrighted)；USGS自产数值数据美国publicdomain/freeuse、保来源署名，不捏造全球CC0或第三方媒体许可。原net/sources/aliases/vintage留私有清单，不抓第三方地图/照片/logo；本地图像不是USGS原图。API/字段来源：[查询API](https://earthquake.usgs.gov/fdsnws/event/1/)、[GeoJSON](https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php)。未新增网络研究/抓取。

root审pub后唯一with_qualified_overlay.py启动本once；旧合格overlay只提供单命令环境，不改共享runtime。新建独占RUN，singleepoch120含准入，Job实际2CPU/2GiB，GeoJSON IPC修复helper原样。任何失败/timeout保原分母与进度，不自动重建。source/rawlabels/media不入Git。
