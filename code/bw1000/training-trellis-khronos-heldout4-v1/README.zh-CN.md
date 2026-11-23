# TRELLIS.2 公开几何留出参考

四个对象在读取几何数据前确定。Avocado、Lantern、FlightHelmet、ToyCar 分别覆盖农产品、便携灯具、防护装备和玩具车辆，与已封存的 WaterBottle、Corset、BoomBox 划分分开。对象名和外部几何缓冲区 SHA-256 均做去重。模型预训练数据暴露情况未知。

`SOURCE_INPUTS.json` 固定每个 glTF、外部几何缓冲区及权利声明的来源、字节数、SHA-256 和采样种子。只需要这些文件，纹理和材质渲染资源不参与几何评分。权利声明对模型内容标记 CC0-1.0，对权利声明元数据标记 CC-BY-4.0，商标和徽标见各资产声明。 [Avocado](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Avocado/LICENSE.md) · [Lantern](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Lantern/LICENSE.md) · [FlightHelmet](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/FlightHelmet/LICENSE.md) · [ToyCar](https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/ToyCar/LICENSE.md)

将权利声明保存为 `<对象>-LICENSE.md`，glTF 保存为 `<对象>-<对象>.gltf`，几何文件保存为 `<对象>.bin` 后，用同一仓库中已固定的 `mesh_quality.py` 构建：

```shell
python -B code/bw1000/training-trellis-khronos-heldout4-v1/build.py \
  --metadata <元数据目录> --buffers <几何目录> --output <新输出目录>
```

以父子节点的全局变换还原场景，保留索引三角面和原图元范围。参考点和未来预测使用同一个以参考包围盒中心及最长边定义的 Y-up 坐标系。每个对象按面积采样 2,048 个参考点，零面积三角面保留在原始分母内。glTF 的 POSITION 边界在访问器声明的 float32 精度下精确核对。 [glTF 2.0 Accessor Bounds](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#accessors-bounds)

输出包含逐对象失败记录。当前四项均生成了参考数据，正式 A100 对照、模型预测、Chamfer-L1 与 F-score@1% 待实测。几何原件和 NPZ 保存在本地忽略目录。

## 配对评分

`score.py` 读取每个对象的同一条件输入、坐标配准和采样种子下生成的三份点云。输入 manifest 使用 `TRELLIS_PUBLIC_HELDOUT4_PAIRED_PREDICTIONS_V1`，`frame` 为 `reference-normalized-glTF-Yup-v1`。顶层的 `checkpoints` 固定 `initial`、`hygon`、`a100` 三组的 `SHA256` 和 `effective_updates`。两组训练后检查点使用相同有效更新数。每个 `objects` 条目固定 `seed`、`conditioning_SHA256` 和 `registration_SHA256`。在 `predictions` 中按组登记这个绑定、`frame`、`checkpoint_SHA256`、`receipt_SHA256` 及 `points_SHA256`。点云分别保存为 `<对象>-<组>.npz`，每份固定为连续 float32 的 2,048 × 3 `points`。

```shell
python -B code/bw1000/training-trellis-khronos-heldout4-v1/score.py \
  --reference-root <参考目录> --prediction-root <预测目录> \
  --manifest <预测清单> --manifest-SHA256 <SHA256> --output <新结果文件>
```

评分保留四对象、十二条预测的固定分母。该程序核对传入摘要与配对绑定，实际模型导出、检查点来源与配准仍由独立回执核验。
