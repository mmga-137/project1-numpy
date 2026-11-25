# 更新仓库

仓库地址：https://github.com/mmga-137/project1-numpy

本地目录已经关联 `origin/main`。修改后在项目根目录执行：

```powershell
git status
git add .
git diff --cached --stat
git commit -m "更新报告和运行说明"
git push origin main
```

提交前看一下文件列表。`weights/`、`assets/` 和生成的 `.npz` 已被忽略，没有修改时不用提交。首次推送按浏览器提示登录 GitHub。

如果修改报告，先用 XeLaTeX 重新编译，再同时提交 `report/main.tex` 和 `report/report.pdf`。

## 在另一台电脑运行

```powershell
git clone https://github.com/mmga-137/project1-numpy.git
cd project1-numpy
python -m pip install -r requirements.txt
```

从课程原附件复制完整的 `weights/` 和 `assets/`，之后按 README 运行。克隆后已经有远程地址，不用再添加 `origin`。

## 使用 GitHub Desktop 或网页

GitHub Desktop 中用 **File → Add local repository…** 选择项目目录。修改后在 **Changes** 中填写说明，点击 **Commit to main → Push origin**。

也可以在仓库网页选择 **Add file → Upload files**，上传要修改的文件后提交。网页上传不会自动按本机 `.gitignore` 过滤文件，要自己确认没有拖入权重和参考数组。通过网页改过文件后，继续本地修改前先同步远程。

当前仓库公开，老师和助教可以直接访问。如果以后改为私有，可在 **Settings → Collaborators → Add people** 中添加他们的账号。课程平台需要的仓库链接和报告还要按要求提交。
