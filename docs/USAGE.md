# 使用说明

## 配准模式

- **同倍率变化**：用于同一物理视野的前后对比。优先读取 TIFF 内嵌或旁路 HDR 中的倍率和 PixelSizeX/Y；缺失时可能从文件名推断倍率，需要核对。
- **跨倍率定位**：将高倍局部定位到低倍视野。按较粗物理网格对齐，不能直接作为定量像素变化分析。
- **人工辅助配准**：至少 6 对拟合点与 3 对独立检查点。点应分散并位于未发生变化的纹理；检查点不参与拟合。

一般导入流程的竖向 SEM 信息栏裁剪使用几何规则，可能不适用于所有仪器版式。批处理已去栏图片时应选择“已经裁掉底栏”。请始终复核裁剪范围与元数据。

普通匹配使用 SIFT，不可用时回退到 ORB；RANSAC 估计局部仿射，再尝试 ECC 精修。ECC 若使匹配几何残差恶化会被拒绝。匹配不足不代表两张图没有对应关系。

自动质量门槛作用于尺度统一后的目标网格，不等同于原始成像分辨率上的精度。报告中的“优秀/良好”仍应结合棋盘格、叠加图或独立检查点复核。

## 标签模式

| 模式 | 保存值 | 含义 |
|---|---|---|
| 双侧孔隙 | 0、1 | Before / After 分别保存 |
| 变化二分类 | 0、1 | 未变化 / 变化 |
| 变化多分类 | 0、1、2、3、4、255 | 未变化、新增孔隙、孔隙扩大、孔隙缩小或闭合、新增裂缝、不确定 |

多分类标注的是变化区域本身，不一定是整个孔隙对象。各模式独立维护标注和撤销记录。PNG 中二值标签保存为 0/1；普通看图软件中可能接近全黑，不代表标签为空。

## 结果与任务恢复

每次新配准建立独立任务目录，主要包括：

```text
before_core.png / after_registered_core.png
difference_core.png / change_mask_initial.png
overlap_mask.png
registration_checkerboard.png / registration_overlay.png
registration_metrics.json / transform_after_to_before.txt
```

保存标注后，另有 `pore_masks/`、`change_binary/`、`change_multiclass/` 等目录。初始空白 mask 表示尚未标注，不能直接用于训练。

可通过“打开已有任务”选择任务目录，或命令行载入索引：

```powershell
python mask_editor_app.py --index "D:\sem-workspace\annotation_index.csv"
```

## 批处理

输入目录使用 `before/after` 或界面支持的 `SEM（before）/SEM（after）` 结构。扫描只是生成候选，文件名相同不等于同一物理视野；用户确认后才进入 GUI 批处理。

已配准结果质检不会重新拟合图像，应输入本程序结果根目录，或 `before/after` 下相对路径同名、尺寸相同的图像。质检报告只能表示自动初检通过、需复核或证据不足。

`build_named_sem_dataset.py` 是按特定 SEM 样品命名规则建立的高级工具，并非任意文件夹的通用导入器。使用前阅读源码中的目录与仪器字段约定。明确指定输入输出：

```powershell
python build_named_sem_dataset.py --source "D:\sem-input" --output "D:\sem-workspace\named_dataset"
python verify_named_dataset.py --output "D:\sem-workspace\named_dataset"
```

## 可选增强匹配

按 [LightGlue 官方文档](https://github.com/cvg/LightGlue) 安装 PyTorch、torchvision 及 LightGlue；这些不是基础编辑器的必需依赖。该模块目前固定使用 CPU，首次运行可能下载匹配权重。实际提升需按图对验证，程序不会保证其优于 SIFT。

## 发布规则

`.gitignore` 默认仅允许源码、Markdown 文档、依赖文件、启动脚本及 CI 配置；图片、实验 CSV/JSON、压缩包、权重和生成目录不进入提交。不要用 `git add -f` 绕过这些规则来加入实验产物。

本次发布不提供实验图片、预训练权重或真实实验的性能承诺。未来添加示例时，应继续使用合成或明确获准公开的数据。
