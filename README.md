# 小项目一：用 NumPy 实现图像特征提取

GitHub 仓库：https://github.com/mmga-137/project1-numpy

直接查看：[实验报告 PDF](report/report.pdf) · [LaTeX 源码](report/main.tex) · [教师检查日志](results/check_all.txt) · [教师数值对照](results/reference_comparison.json) · [仓库检查记录](docs/REPOSITORY_AUDIT.md)

Python 程序设计课程作业，报告人：胡楷。使用课程提供的 OpenCLIP ViT-B-32 图像参数，以 NumPy 完成前向计算，将每张图片转换为长度为 512 的单位特征向量。无需训练、GPU、PyTorch 或 OpenCLIP 软件包。

## 功能与实现

正式实现位于 `image_encoder.py`，已补全全部六组 TODO：

| 任务 | 方法 | 形状变化或计算 |
| --- | --- | --- |
| TODO-01 | `ImageEncoder.patchify` | `[B,3,224,224] → [B,49,3072]`；块从左到右、从上到下排列，块内按 C、H、W 展开 |
| TODO-02 | `MultiHeadAttention.split_heads` | `[B,N,D] → [B,H,N,D/H]`，图像模型为 12 个头，每头 64 维 |
| TODO-03 | `MultiHeadAttention.__call__` | 缩放点积注意力、V 加权、多头合并与输出投影 |
| TODO-04 | `TransformerBlock.__call__` | 两次残差连接及 `768 → 3072 → 768` 的 GELU MLP |
| TODO-05 | `ImageEncoder.embed` | 小块投影，拼接 CLS，加入学习的位置编码 |
| TODO-06 | `ImageEncoder.__call__` | 取 CLS，post-Norm、投影到 512 维，再单位化 |

所有模型参数保持固定。预处理、数值辅助函数、资源加载器和教师检查脚本沿用课程文件。最终特征表示图片内容；本项目没有加入分类器或文本编码器。

## 文件结构

```text
project1-numpy/
├── image_encoder.py          # 补全后的编码器
├── helpers.py                # 课程提供的 Norm、GELU 等
├── preprocess_image.py       # 课程提供的图像预处理
├── runtime.py                # 参数和图片加载
├── run.py                    # 推理入口
├── check.py                  # 课程分阶段检查
├── config.json               # 固定模型配置
├── requirements.txt          # 依赖范围
├── SOURCE-MANIFEST.json      # 原始课程附件来源与指纹记录
├── tests/
│   ├── test_image_encoder.py  # 不需要课程数据的单元测试
│   ├── verify_inference.py    # 新输入、单张/批量和独立命令行验证
│   └── compare_reference.py   # 与教师逐元素比较并导出报告用的对照值
├── docs/
│   ├── COURSE_README.md      # 课程原始说明，供追溯
│   ├── debug_patch_order.py  # 切块顺序错误的可复现实验
│   ├── REVIEW.md             # 提交前复查发现与修复
│   ├── REPORT_REVISION.md    # 报告修订与数值核对记录
│   ├── REPOSITORY_AUDIT.md   # GitHub 仓库与干净副本复现检查
│   └── UPLOAD_GITHUB.md      # GitHub 上传步骤
├── results/                  # 实际运行日志、环境和输出摘要
├── report/                   # 报告 LaTeX 源码和 PDF
├── weights/                  # 从课程附件恢复；不上传
└── assets/                   # 从课程附件恢复；不上传
```

`SOURCE-MANIFEST.json` 是原始附件的来源记录，不是修改后文件的完整性清单。原始说明中提到的课堂演示、练习题和 PPT 映射未复制进本提交目录；需要时请查看课程原附件。

## 环境安装

使用 Python 3.9 或以上版本，在仓库根目录运行：

```powershell
python -m pip install -r requirements.txt
```

已验证环境和检查时间详见 `results/environment.json`。浮点结果可能随 NumPy 与底层矩阵库版本略有变化，应以教师检查脚本的误差阈值判定。

## 恢复课程数据与参数

课程 README 要求不重复提交大型参数与参考数据，因此 `.gitignore` 排除了 `weights/`、`assets/` 和生成的 `.npz` 文件。

请从课程平台下载老师提供的 **project1-numpy-student** 原始附件，解压后将其中的 **整个 `weights` 文件夹和整个 `assets` 文件夹**复制到本仓库根目录。仓库没有为这些课程资源提供公开下载地址。老师和助教可使用原始附件中的同一组文件。

恢复后至少应存在：

```text
weights/image_encoder.npz
weights/manifest.json
assets/reference_outputs.npz
assets/manifest.json
assets/images.json
assets/images/image_01.jpg  ...  image_10.jpg
```

如果本机仍有原始附件，也可以在 PowerShell 中执行，按实际解压位置修改路径：

```powershell
Copy-Item -LiteralPath "C:\Users\HuKai\Downloads\project1-numpy-student\weights" -Destination . -Recurse
Copy-Item -LiteralPath "C:\Users\HuKai\Downloads\project1-numpy-student\assets" -Destination . -Recurse
```

本次交付的本地工作目录已放好课程资源，可以直接运行；GitHub 仓库和提交压缩包不包含这些资源。

## 运行与验证

完整教师检查：

```powershell
python check.py --stage all
```

也可按 `assets → preprocess → patches → tokens → block0 → last_tokens → output → batch` 的顺序逐个检查，例如 `python check.py --stage block0`。定位错误时先修复最早失败的阶段。

处理课程提供的十张图片：

```powershell
python run.py --output output_tokens.npz
```

处理其他图片（仍需课程模型参数）：

```powershell
python run.py --images "path/to/a.jpg" "path/to/b.png" --output my_tokens.npz
```

输出 `.npz` 包含 `ids` 与 `output_tokens` 两个数组。特征为有限的 float32 数组 `[B,512]`，顺序与输入一致，每行的 L2 长度约为 1。

不依赖课程数据的补充检查和排错实验：

```powershell
python -m unittest discover -s tests -v
python docs/debug_patch_order.py
```

六个单元测试使用合成数组，覆盖输入形状和类型检查、非连续内存的切块顺序、通用的多头拆分，以及与独立逐头循环实现的注意力比较（包括 causal 模式）。编码器要求输入为 `[B,3,224,224]` 的 float32 NumPy 数组，形状不匹配时抛出 `ValueError`，类型不匹配时抛出 `TypeError`。传入图片路径时，`run.py` 会调用课程预处理生成合法输入。

恢复课程权重后，还可执行独立推理验证：

```powershell
python tests/verify_inference.py --output results/extra_validation.json
```

该脚本使用新生成的输入检查单张和批量输出的一致性，并建立只含代码和权重的临时项目，不放入 `assets/`。它从其他工作目录调用命令行，处理新生成的 RGB、灰度图片，将结果与直接调用编码器的输出比较。详细检查记录保存在 `results/extra_validation.json`，不依赖教师参考数组。

恢复完整课程资源后，可导出逐阶段容差及本程序与教师的具体数值对照：

```powershell
python tests/compare_reference.py --output results/reference_comparison.json
```

其中每个元素的允许误差为 `atol + rtol * abs(reference)`；误差与允许值之比不超过 1 才算符合要求。`results/reference_comparison.json` 保存各阶段的最大比值及四个输出位置的实际值、教师参考值、绝对误差和允许误差。位置选择包括首值、最大绝对误差处、参考值最接近零处及末值；特征下标从 0 开始。

## 本次运行结果

教师八个阶段全部通过，原始输出保存在 `results/check_all.txt`：

| 阶段 | 输出形状 | 最大绝对误差 | 平均绝对误差 |
| --- | --- | --- | --- |
| preprocess | `(10,3,224,224)` | `0` | `0` |
| patches | `(2,49,3072)` | `0` | `0` |
| tokens | `(2,50,768)` | `4.387e-05` | `1.335e-06` |
| block0 | `(2,50,768)` | `7.147e-06` | `3.348e-07` |
| last_tokens | `(2,50,768)` | `4.196e-05` | `2.633e-06` |
| output | `(10,512)` | `5.662e-07` | `8.688e-08` |
| batch | `(3,512)` | `4.061e-07` | `7.848e-08` |

`assets` 阶段通过 152 个视觉参数张量与十组图片/参考输出的完整性检查。检查逐元素使用 `abs(actual-reference) <= atol + rtol*abs(reference)`；最终输出的 `atol=2e-5, rtol=2e-4`。除误差外，还检查形状、float32、有限值和输出单位长度。

`results/output_summary.json` 记录课程十张图片实际输出的形状与长度范围；`results/unit_tests.txt` 记录六个单元测试通过。新增验证中，单张与四张换序/重复 batch 的最大绝对差为 `1.490e-07`；没有 `assets/` 的独立项目完成两张新图推理，命令行与 API 输出的最大差为 `0`。这些额外检查验证批处理和资源独立性；模型数值准确性仍由教师参考检查判定。

逐元素对照还确认：最终 5120 个特征值均未超出允许范围，最大误差/允许误差比约为 `0.02053`。最大绝对误差在 `image_10` 的下标 `152`：本程序输出 `0.11359964311122894`，教师参考 `0.11360020935535431`，绝对误差约 `5.662e-07`，小于该元素允许的 `4.272e-05`。最大绝对误差位置与最大比值位置不一定相同。

## 形状变化与排错案例

复查中发现一个真实问题：原实现只根据元素总数 reshape，会把 `(1,3,112,448)` 或 NHWC 格式的数组错误地解释为固定尺寸 CHW 图像，也会接受 float64 输入。现在切块前显式检查数组类型、四维形状和 float32 类型，并补充回归测试，避免把输入问题拖到后续矩阵计算或数值检查中。具体记录见 `docs/REVIEW.md`。

切块分为三步：`[B,3,224,224] → [B,3,7,32,7,32] → [B,7,7,3,32,32] → [B,49,3072]`。直接 `reshape(B,49,3072)` 虽然形状相同，但会把整幅图的连续像素误当成一个方形图片块。

`docs/debug_patch_order.py` 专门构造了这一错误作为排错对照，用逐块切片建立独立期望值。错误版本有 150464 个元素位置不匹配，采用轴重排后的正式实现为 0。该记录是可复现的诊断实验，报告按实验事实描述。

其他关键约定：注意力对 K 只交换最后两个轴；普通线性层使用 `weight.T`；最终 `visual.proj` 已为 `[768,512]`，直接相乘；第一条残差加回原始输入，第二条加回第一次残差结果。

## 报告与辅助使用说明

`report/main.tex` 和 `report/report.pdf` 是依据代码、实际日志与学习讨论整理的实验报告，正文 5 页，另含封面和批阅页，总计 7 页。封面沿用报告人此前指定的信息和日期；各次验证的实际时间见 `results/` 中对应的环境和对照记录。使用 XeLaTeX 编译 `report/main.tex`。报告中已填写仓库地址；提交前请本人阅读并核对报告内容和封面日期。

课程教师提供任务框架、固定参数、样例图片、数值参考、辅助函数、运行与检查脚本。胡楷的参与包括：

- **图像切块与索引分析**：推演小矩阵示例，核对块的空间位置、扫描顺序和块内展开顺序，辨析直接 reshape 的局限。
- **块投影原理分析**：梳理 3072 维像素向量到 768 维特征的加权组合，区分特征维度变化、像素分组与图片缩放。
- **局部到全图的流程梳理**：明确位置编码在 Transformer 前加入，分析 Transformer 信息融合及 CLS 汇总的作用和先后关系。
- **多头注意力分析**：讨论 Q、K、V 的作用，梳理 768 维拆成 12 个头、每头 64 维的关系，确认每头仍可关注全部 token。
- **残差计算推演**：完成两次相加的数值示例，区分第一条残差加回原输入、第二条加回第一次更新结果的关系。
- **MLP 与 GELU 分析**：理解 `768→3072→768` 的特征加工和非线性激活作用，明确 token 数量保持不变。
- **LayerNorm 作用分析**：围绕数值分布调整提出疑问，讨论尺度控制、后续计算稳定性及归一化的作用。
- **验证思路梳理**：提出补充教师参考值直接对照的要求，梳理绝对误差、允许误差和逐元素验收判据。

其他参与包括课程任务与项目资料整理、报告信息及版式要求确定、GitHub 仓库创建与授权，以及代码复查和文档修改要求的提出。

本项目使用 AI 辅助学习：围绕不理解的概念讨论，再通过切块、残差的小例子推演，将理解反馈给 AI 核对。代码实现、数值检查、报告整理和 GitHub 提交也有 AI 协助。辅助工具：OpenAI Codex。

## 上传 GitHub

操作步骤见 `docs/UPLOAD_GITHUB.md`。上传本仓库的源码、说明、日志和报告即可，课程资源由 `.gitignore` 排除。保留实际的提交记录，并给老师、助教开放访问权限。
