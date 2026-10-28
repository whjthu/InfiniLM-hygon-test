# r162a：修复必要加载的提交前队列校验

旧 r162 的实际 CPU keeper 910407.515 在 validate 阶段以 KeyError GSM_unique 失败，GPU_submitted=false，groups=[]，没有开始模型加载。旧结果和 namespace 保留。

本包只将所有运行/结果/源码路径改为独立 r162a，并将陈旧的 GSM32/12case/no_new_tokenization guard 改为实际 frozen queue 校验。keeper 与 CPU source contract 共用 pilot_queue_contract.py：6case，C1四题+C8两组，20次执行、16独立题、C1subset，原 cap512/context1024/W8/state/W2pairON/cacheON/W13OFF。CPU source contract 仍逐题与原冻结题目/答案比较，源码SHA冻结后由 keeper 验证。未删除生命周期/网络/内存/capacity/质量条件。

复用 CPU511 Native798c4bca01cddb0755a91171ef61dfa755329518bb8cfc87640305a24e55a6e1，不重新编译、不添加GPU优化。固定ON加载只能归因当前加载工作，不能声称旧冷加载的因果加速。没有SSH/Git/Slurm，没有实际GO。实际部署目录 /refactor/incoming/Kimi-sparse-profile-Model-r162a/，入口 launcher/preflight_then_run.py，root审阅发布后生成对应GO并执行。
