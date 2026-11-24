# 将本项目上传到 GitHub

本地目录：`G:\ChatGPT\Python项目\project1-numpy`。

目录已经初始化为独立 Git 仓库，分支为 `main`，远程 `origin` 为 https://github.com/mmga-137/project1-numpy.git。已有 GitHub 仓库的初始化记录会保留，之后每次修改正常提交即可。

上述初始化状态适用于原本地目录；提交 ZIP 不含 `.git`。如果使用 ZIP 解压出的新副本，GitHub Desktop 提示目录不是 Git 仓库时，点击 **create a repository here**，核对创建位置就是该副本，再继续提交与发布。

## 推荐：GitHub Desktop

1. 安装并打开 GitHub Desktop（https://desktop.github.com/），登录自己的 GitHub 账号。若提示配置 Git 作者，填写你希望显示的名称与 GitHub 账号对应的邮箱；也可使用 GitHub 提供的 noreply 邮箱。
2. 点击 **File → Add local repository…**，选择上述 `project1-numpy` 文件夹，点击 **Add repository**。
3. 在 **Changes** 中查看文件。应能看到 `image_encoder.py`、`README.md`、`report/` 等；`weights/`、`assets/`、`output_tokens.npz` 已忽略。
4. 如果 **Changes** 中还有待提交的修改，在左下角 **Summary** 填写修改说明，点击 **Commit to main**；没有待提交修改时可直接同步远程。
5. 本目录已有远程地址，点击 **Push origin** 推送新提交，或 **Fetch origin** 检查同步状态。如果从 ZIP 副本新建自己的仓库，可使用 **Publish repository**，名称填 `project1-numpy`。
6. 点击 **View on GitHub** 打开仓库。当前给定仓库可公开访问；若以后设为私有，请进入 **Settings → Collaborators**（界面也可能显示 **Collaborators and teams**），点击 **Add people** 添加老师和助教账号。邀请被接受后才能访问私有仓库。
7. 复制浏览器中的仓库地址，写进实验报告或课程平台指定位置。不要把 GitHub 个人主页地址当作仓库地址。

仓库发布以后，如果修改了报告或代码，在 GitHub Desktop 中再次 **Commit to main → Push origin**，GitHub 才会更新。

## 备选：Git 命令行

先在 https://github.com/new 新建空仓库，名称 `project1-numpy`，可设为 Private，**不要勾选自动添加 README、.gitignore 或 License**。随后安装 Git for Windows（若本机尚无可用 Git），重开 PowerShell：

```powershell
cd "G:\ChatGPT\Python项目\project1-numpy"
git config user.name "你的GitHub显示名称"
git config user.email "你的GitHub邮箱或noreply邮箱"
git add .
git status
git commit -m "Complete NumPy image encoder and add validation report"
git remote add origin https://github.com/你的用户名/project1-numpy.git
git push -u origin main
```

把名称、邮箱和远程地址替换成自己的值。身份验证按 Git 的浏览器登录提示完成。如果仓库中已设置 `origin`，先用 `git remote -v` 核对，不要重复添加。已有远程提交的仓库请正常合并，不要强制推送覆盖。

## 仅使用网页上传

若暂时不使用 Git 工具，可解压交付的 `project1-numpy-github.zip`，进入 GitHub 新仓库的 **uploading an existing file**，或已有仓库的 **Add file → Upload files**，上传解压目录中的文件和子文件夹，然后填写说明并 **Commit changes**。

请上传源码、说明、日志与报告本身；整个 ZIP 作为唯一文件上传不方便老师查看代码。压缩包已经排除了课程大文件，网页上传不会自动应用本机 `.gitignore`。

## 提交前核对

- 仓库首页可直接看到 README 和 `image_encoder.py`。
- `results/check_all.txt`、环境记录和报告已上传。
- 老师、助教能访问仓库；私有仓库的访问邀请已接受。
- 报告已由本人核对，课程平台中的仓库链接指向正确仓库。
- README 已说明从课程原附件恢复完整 `weights/` 与 `assets/` 的方法。
- 没有上传 326 MB 模型参数或参考数组。

本项目保留了实际运行日志和辅助使用说明。GitHub 上的首次提交日期以真正上传/提交时为准，不需要补造此前的记录。
