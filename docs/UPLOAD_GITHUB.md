# 同步本项目到 GitHub

仓库：https://github.com/mmga-137/project1-numpy

本地目录：`G:\ChatGPT\Python项目\project1-numpy`。本目录的 `main` 分支已关联该仓库的 `origin/main`，可以直接提交后续修改。

## 更新已经关联的本地目录

PowerShell 中执行：

```powershell
cd "G:\ChatGPT\Python项目\project1-numpy"
git status
git add .
git diff --cached --stat
git commit -m "Describe the changes made in this update"
git push origin main
```

提交说明应描述本次实际修改。`git diff --cached --stat` 用于核对即将上传的文件；`weights/`、`assets/` 和生成的 `.npz` 已被忽略。没有文件变化时无需创建新提交。首次身份验证按浏览器提示由本人完成。

使用 GitHub Desktop 时，选择 **File → Add local repository…** 添加上述目录；修改后在 **Changes** 中填写说明，点击 **Commit to main → Push origin**。点击 **View on GitHub** 查看结果。

## 在另一台电脑取得项目

```powershell
git clone https://github.com/mmga-137/project1-numpy.git
cd project1-numpy
python -m pip install -r requirements.txt
```

从课程原附件复制完整 `weights/` 与 `assets/`，按 README 运行。克隆得到的目录已经配置好远程地址。

## 网页上传或更新

也可以在 GitHub 仓库选择 **Add file → Upload files**，上传需要更新的源码、说明、日志或报告，再填写说明并 **Commit changes**。提交 ZIP 已排除大文件；使用网页上传时，本机 `.gitignore` 不会自动过滤拖入的文件。

报告有修改时，请同时更新 `report/main.tex` 和重新编译的 `report/report.pdf`。如果通过网页编辑了文件，在继续本地修改前先同步远程；请保留双方记录，不要强制推送覆盖。

## 提交核对

- 仓库根目录可直接看到 README 和 `image_encoder.py`。
- `results/check_all.txt`、环境记录、额外验证和报告均已同步。
- 在课程平台提交正确的仓库链接。
- README 已说明从课程原附件恢复数据和权重的办法。
- 本人已阅读报告、核对日期与辅助使用说明。
- 当前仓库可公开访问；若以后改为私有，在 **Settings → Collaborators → Add people** 邀请老师和助教，并等待对方接受。

GitHub 提交记录使用实际操作日期，按每次真实修改提交即可。
