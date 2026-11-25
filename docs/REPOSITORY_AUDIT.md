# 仓库检查记录

检查日期：2026/11/25。被检查的代码和报告版本为 [29845ae](https://github.com/mmga-137/project1-numpy/commit/29845ae6c80ca1a5381bbe8bb0bd40058d01665c)。之后只调整了封面日期和文档，没有改算法。

| 项目 | 检查结果 |
| --- | --- |
| 访问 | 仓库公开，默认分支为 `main`，未归档 |
| 文件 | 源码、依赖、配置、README、测试日志和报告齐全，远程文件哈希与被检查提交一致 |
| 大文件 | 未提交权重、参考数组、缓存或编译临时文件；该版本最大文件为 PDF，310668 字节，历史最大文件为 318571 字节 |
| 实现 | 六组 TODO 已补全；编码器不读取参考数组；课程辅助文件保持原有内容 |
| 基础测试 | 干净副本中未放课程资源时，六个单元测试和切块示例通过 |
| 运行 | 恢复课程资源后，教师八阶段检查、十图推理、数值对照和独立新图推理都通过 |
| 误差 | 最终输出 `(10,512)`、float32，最大绝对误差约 `5.662e-07`，所有比较元素都在容差内 |
| 报告 | 包含任务、方法、结果对照、问题修复、反思、八项技术参与和辅助说明；正文 5 页，总计 7 页 |
| 源码与 PDF | 被检查版本重新编译后，七页渲染图与上传 PDF 逐页像素一致 |

这次使用的是提交导出的干净副本。先执行不需要课程资源的检查：

```powershell
python -m unittest discover -s tests -v
python docs/debug_patch_order.py
```

按 README 恢复整个 `weights/` 和 `assets/` 后执行：

```powershell
python check.py --stage all
python run.py --output output_tokens.npz
python tests/compare_reference.py --output results/reference_comparison.json
python tests/verify_inference.py --output results/extra_validation.json
```

没有发现影响课程提交的问题。新下载仓库后，需要先补齐课程资源再运行全量检查；大型参数和参考数据不上传是课程原说明的要求。
