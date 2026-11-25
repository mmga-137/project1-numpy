# 小项目一：用 NumPy 实现图像特征提取

GitHub 仓库：https://github.com/mmga-137/project1-numpy

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
│   └── verify_inference.py    # 新输入、单张/批量和独立命令行验证
├── docs/
│   ├── COURSE_README.md      # 课程原始说明，供追溯
│   ├── debug_patch_order.py  # 切块顺序错误的可复现实验
│   ├── REVIEW.md             # 提交前复查发现与修复
│   └── UPLOAD_GITHUB.md      # GitHub 上传步骤
├── results/                  # 实际运行日志、环境和输出摘要
├── report/                   # 报告 LaTeX 源码和 PDF 初稿
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

## 形状变化与排错案例

复查中发现一个真实问题：原实现只根据元素总数 reshape，会把 `(1,3,112,448)` 或 NHWC 格式的数组错误地解释为固定尺寸 CHW 图像，也会接受 float64 输入。现在切块前显式检查数组类型、四维形状和 float32 类型，并补充回归测试，避免把输入问题拖到后续矩阵计算或数值检查中。具体记录见 `docs/REVIEW.md`。

切块分为三步：`[B,3,224,224] → [B,3,7,32,7,32] → [B,7,7,3,32,32] → [B,49,3072]`。直接 `reshape(B,49,3072)` 虽然形状相同，但会把整幅图的连续像素误当成一个方形图片块。

`docs/debug_patch_order.py` 专门构造了这一错误作为排错对照，用逐块切片建立独立期望值。错误版本有 150464 个元素位置不匹配，采用轴重排后的正式实现为 0。该记录是可复现的诊断实验，报告按实验事实描述。

其他关键约定：注意力对 K 只交换最后两个轴；普通线性层使用 `weight.T`；最终 `visual.proj` 已为 `[768,512]`，直接相乘；第一条残差加回原始输入，第二条加回第一次残差结果。

## 报告与辅助使用说明

`report/main.tex` 和 `report/report.pdf` 是依据代码与实际日志整理的报告初稿。封面沿用报告人此前指定的信息和日期；验证记录的实际时间以 `results/environment.json` 为准。使用 XeLaTeX 编译 `report/main.tex`。报告中已填写仓库地址；提交前请本人阅读并核对报告内容和封面日期。

课程教师提供任务框架、固定参数、样例图片、数值参考、辅助函数、运行与检查脚本。胡楷的参与包括：

- 整理课程任务、项目附件及实验报告模板，明确交付内容与封面排版要求。
- 创建 GitHub 仓库、完成账号授权，提出代码复查与文档修改要求。
- 推演小矩阵的图像切块示例，理解扫描顺序、块内展开顺序及直接 reshape 的局限。
- 梳理块投影中 3072 维像素向量到 768 维特征的变化，辨析加权组合与简单缩小维度的区别。
- 梳理位置编码、Transformer 信息融合和 CLS 汇总的先后关系，解释局部特征到全图特征的流程。
- 参与 Q、K、V 和多头注意力原理讨论，确认每个头仍可关注全部 token。
- 完成两次残差相加的数值示例推演，区分两条残差路径的相加对象。
- 参与 MLP、GELU 与 LayerNorm 的概念分析，讨论特征维度变化、非线性加工和数值尺度控制的作用。

辅助工具：OpenAI Codex。

## 上传 GitHub

操作步骤见 `docs/UPLOAD_GITHUB.md`。上传本仓库的源码、说明、日志和报告即可，课程资源由 `.gitignore` 排除。保留实际的提交记录，并给老师、助教开放访问权限。
