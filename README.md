# 小项目一：NumPy 图像特征提取

Python 程序设计课程作业，胡楷。

根据课程给出的 OpenCLIP ViT-B-32 参数，用 NumPy 补全图像编码器的前向计算。输入图片后，程序为每张图片输出一个 512 维单位特征向量。参数已经给定，运行时不需要训练或使用 GPU。

[实验报告 PDF](report/report.pdf) · [报告源码](report/main.tex) · [教师检查日志](results/check_all.txt) · [数值对照](results/reference_comparison.json)

## 实现内容

六组 TODO 都在 `image_encoder.py` 中：

| 任务 | 方法 | 形状变化或计算 |
| --- | --- | --- |
| TODO-01 | `ImageEncoder.patchify` | `[B,3,224,224] → [B,49,3072]`；块从左到右、从上到下排列，块内按 C、H、W 展开 |
| TODO-02 | `MultiHeadAttention.split_heads` | `[B,N,D] → [B,H,N,D/H]`，图像模型为 12 个头，每头 64 维 |
| TODO-03 | `MultiHeadAttention.__call__` | 缩放点积注意力、V 加权、多头合并与输出投影 |
| TODO-04 | `TransformerBlock.__call__` | 两次残差连接及 `768 → 3072 → 768` 的 GELU MLP |
| TODO-05 | `ImageEncoder.embed` | 小块投影，拼接 CLS，加入学习的位置编码 |
| TODO-06 | `ImageEncoder.__call__` | 取 CLS，post-Norm、投影到 512 维，再单位化 |

预处理、LayerNorm、GELU、资源加载和教师检查脚本使用课程提供的版本。这个项目完成的是图像编码器，没有加入分类或文本编码功能。

## 文件结构

```text
project1-numpy/
├── image_encoder.py          # 补全后的编码器
├── helpers.py                # LayerNorm、GELU 等课程辅助代码
├── preprocess_image.py       # 图片预处理
├── runtime.py                # 加载参数和图片
├── run.py                    # 运行入口
├── check.py                  # 教师分阶段检查
├── config.json               # 模型配置
├── requirements.txt          # 依赖清单
├── SOURCE-MANIFEST.json      # 原始课程附件的来源记录
├── tests/
│   ├── test_image_encoder.py # 小数组单元测试
│   ├── verify_inference.py   # 新图片和独立运行检查
│   └── compare_reference.py  # 导出教师数值对照
├── docs/                     # 原始课程说明、排错笔记和提交说明
├── results/                  # 测试日志、环境信息和结果摘要
├── report/                   # 实验报告 main.tex 和 report.pdf
├── weights/                  # 从课程附件复制，不上传
└── assets/                   # 从课程附件复制，不上传
```

`SOURCE-MANIFEST.json` 保留的是原始附件的信息。课堂演示和练习题仍在课程原附件中，本仓库只放完成项目需要的代码和说明。

## 环境和数据准备

需要 Python 3.9 或以上版本。下载仓库并安装依赖：

```powershell
git clone https://github.com/mmga-137/project1-numpy.git
cd project1-numpy
python -m pip install -r requirements.txt
```

本次运行使用 Python 3.13.12、NumPy 2.4.2 和 Pillow 12.1.1，完整环境记录在 [environment.json](results/environment.json)。

从课程平台下载 **project1-numpy-student** 原始附件，把解压后的**整个 `weights` 文件夹和整个 `assets` 文件夹**复制到仓库根目录。课程要求不重复提交大型参数和参考数据，因此这两个目录没有上传。

复制后应有以下文件，图片及其余文件也要保留：

```text
weights/image_encoder.npz
weights/manifest.json
assets/reference_outputs.npz
assets/manifest.json
assets/images.json
assets/images/image_01.jpg  ...  image_10.jpg
```

## 运行方法

先检查各阶段的计算结果：

```powershell
python check.py --stage all
```

也可以单独检查一阶段，例如 `python check.py --stage patches`。检查顺序是 `assets → preprocess → patches → tokens → block0 → last_tokens → output → batch`，有问题时从最早失败的阶段查起。

提取课程十张图片的特征：

```powershell
python run.py --output output_tokens.npz
```

换成自己的图片：

```powershell
python run.py --images "path/to/a.jpg" "path/to/b.png" --output my_tokens.npz
```

输出文件包含 `ids` 和 `output_tokens`。特征数组的形状为 `[B,512]`，类型是 float32，每行长度约为 1，排列顺序与输入图片一致。

## 检查结果

教师八个阶段检查全部通过。下面是与参考结果的误差，原始日志在 [check_all.txt](results/check_all.txt)：

| 阶段 | 输出形状 | 最大绝对误差 | 平均绝对误差 |
| --- | --- | --- | --- |
| preprocess | `(10,3,224,224)` | `0` | `0` |
| patches | `(2,49,3072)` | `0` | `0` |
| tokens | `(2,50,768)` | `4.387e-05` | `1.335e-06` |
| block0 | `(2,50,768)` | `7.147e-06` | `3.348e-07` |
| last_tokens | `(2,50,768)` | `4.196e-05` | `2.633e-06` |
| output | `(10,512)` | `5.662e-07` | `8.688e-08` |
| batch | `(3,512)` | `4.061e-07` | `7.848e-08` |

`assets` 检查了 152 个视觉参数张量和十组图片、参考结果。数值检查使用 `abs(actual-reference) <= atol + rtol*abs(reference)`，最终输出的 `atol=2e-5`、`rtol=2e-4`，同时检查形状、类型、有限值和单位长度。

最终 5120 个特征值都在允许范围内，最大的“误差/允许误差”比约为 `0.02053`。最大绝对误差出现在 `image_10` 的下标 `152`：程序输出约 `0.113599643`，教师参考约 `0.113600209`，误差为 `5.662e-07`，小于该位置允许的 `4.272e-05`。完整精度数值见 [reference_comparison.json](results/reference_comparison.json)，下标从 0 开始。

## 补充测试和排错

下面两项使用小数组，不需要课程参数或图片：

```powershell
python -m unittest discover -s tests -v
python docs/debug_patch_order.py
```

六个单元测试检查输入形状和类型、切块顺序、拆头及注意力计算。切块时要先拆开网格轴，再调整顺序，不能直接 `reshape(B,49,3072)`。排错示例中，直接 reshape 有 150464 个元素错位，正确实现为 0。

补齐课程资源后，还可以运行：

```powershell
python tests/verify_inference.py --output results/extra_validation.json
python tests/compare_reference.py --output results/reference_comparison.json
```

新输入单张与四张批量输出的最大差为 `1.490e-07`。另外，只放入代码和权重的临时项目也能处理 RGB、灰度新图，命令行与 API 输出的最大差为 `0`。这些检查验证了批处理和独立运行；本项目没有评测分类或检索准确率。

复查还发现，只按元素总数 reshape 会接受 `(1,3,112,448)` 或 NHWC 等错误输入。现在切块前会检查 `[B,3,224,224]` 和 float32。具体例子见 [实现笔记](docs/REVIEW.md)。

## 本人参与

学习和讨论中，我主要参与了以下内容：

- **图像切块**：用小矩阵推演块的位置、扫描和展开顺序，弄清直接 reshape 为什么会排错。
- **块投影**：理解 `3072→768` 是对整个块做加权组合，分清它与像素分组、图片缩放的区别。
- **特征汇总流程**：厘清位置编码、Transformer 和 CLS 的先后关系，说明局部特征如何汇总成全图特征。
- **多头注意力**：讨论 Q、K、V 的作用，理解 12 个头、每头 64 维，以及每头仍能关注全部 token。
- **残差计算**：手算两次相加的例子，分清两条残差路径各自加回什么。
- **MLP 与 GELU**：讨论 `768→3072→768` 和非线性激活的作用，明确 token 数量没有变化。
- **LayerNorm**：围绕数值分布调整提出疑问，理解控制数值尺度对后续计算的意义。
- **结果核对**：提出补充教师值直接对照的要求，梳理绝对误差、允许误差和逐元素判定方法。

另外参与了项目资料整理、报告要求确定、仓库创建和复查。

使用了 OpenAI Codex 辅助学习：讨论原理后用小例子推演，再反馈自己的理解进行核对。代码实现、数值检查、报告整理和 GitHub 提交也有 AI 协助。

## 报告和其他说明

[实验报告](report/report.pdf)正文 5 页，加封面和批阅页共 7 页，提交日期为 2026/11/25。LaTeX 源码在 `report/main.tex`，使用 XeLaTeX 编译。

- [课程原始说明](docs/COURSE_README.md)
- [实现笔记](docs/REVIEW.md)
- [仓库检查记录](docs/REPOSITORY_AUDIT.md)
- [报告修改记录](docs/REPORT_REVISION.md)
- [更新仓库的方法](docs/UPLOAD_GITHUB.md)
