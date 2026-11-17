# USGS saved-body 独立 reader，准备未消费

仅核固定原v2的intent/outcome/worker-result/progress与35,802B已保存body。原producer KeyError/count、source_accessible=false、Job C1永久保留。原HTTP200与取得原body是独立事实，非producer成功。新reader没有HTTP/URL请求库，不import旧sample/getter；不重GET。

query API2.8.4实际metadata没有count，分母严格len(features)，metadata_count=None/metadata_count_present=false。存在count时严格整数且等于len(features)；limit仅请求上限，不注入count。其他Decimal/aliases/region/null/errors与原audit语义保留。返回50且limit50不能称整窗完整；不生成图/QA/split，不加载模型，不读WDI/CORD等封存源。所有Dataset/training/GPU GO=false。

root审核后唯一 `read_once.py --root-go`；独占新local-only输出私有行摘要和小result。源事件ID/原数值不进Git或stdout。新增schema边界只synthetic，不能冒真实消费。科学来源/权利沿用原官方USGS API/GeoJSON模式/版权FAQ，不改变新场景/数据权利。
