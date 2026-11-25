# GitHub 仓库提交检查

检查日期：2026/11/25。代码与报告的被检查版本为 [29845ae](https://github.com/mmga-137/project1-numpy/commit/29845ae6c80ca1a5381bbe8bb0bd40058d01665c)。本次后续修改仅补充 README 导航和本检查记录。

结论：按课程截图及原始 README 中的提交要求检查，未发现影响提交的问题。

| 检查内容 | 实际结果 |
| --- | --- |
| GitHub 访问与分支 | 仓库公开，默认分支为 `main`，未归档 |
| 远程内容 | 远程文件树的内容哈希逐项与被检查提交核对一致 |
| 提交文件 | 源码、依赖、配置、README、验证记录、报告 PDF 与 LaTeX 源码齐全 |
| 大文件与临时文件 | 未提交课程权重、参考数组、缓存、环境目录或 LaTeX 临时文件；被检查版本最大文件为报告 PDF，310668 字节；历史最大文件为 318571 字节 |
| 实现与课程文件 | 六组 TODO 已实现；编码器不读取参考数组；课程预处理、辅助函数、运行与教师检查文件保持原有内容 |
| 干净副本的基础测试 | 未放入课程资源时，六个单元测试和切块示例通过 |
| 恢复课程资源后复现 | 教师八阶段检查、默认十图命令行推理、逐元素教师对照及独立新图推理均通过 |
| 数值结果 | 最终输出为 `(10,512)`、float32，最大绝对误差约 `5.662e-07`，教师对照中的所有元素均符合容差 |
| 报告内容 | 包含任务、方法、复现步骤、实际误差、具体教师值对照、问题修复、反思、八项技术参与和辅助说明 |
| PDF 与源码一致性 | 从被检查提交取出源码重新编译，七页渲染图与远程版本 PDF 逐页像素一致 |
| 报告篇幅 | 正文 5 页，另有封面和批阅页，总计 7 页，符合截图对正文不超过 5 页的要求 |

复现时使用被检查提交导出的干净文件副本，避免本机未提交文件影响检查。安装依赖后，先执行不需要课程资源的检查：

```powershell
python -m unittest discover -s tests -v
python docs/debug_patch_order.py
```

按 README 从课程原附件恢复完整 `weights/` 和 `assets/` 后执行：

```powershell
python check.py --stage all
python run.py --output output_tokens.npz
python tests/compare_reference.py --output results/reference_comparison.json
python tests/verify_inference.py --output results/extra_validation.json
```

课程要求不重复提交大型参数与参考数据，因此新下载的仓库需要恢复这些资源后才能执行教师全量检查。这是已在 README 说明的复现步骤。

本检查确认的是文件组织、复现、数值和报告一致性；最终成绩仍由教师结合报告和课程考核要求评定。在课程平台提交仓库链接及所需报告，属于仓库之外的提交步骤。
