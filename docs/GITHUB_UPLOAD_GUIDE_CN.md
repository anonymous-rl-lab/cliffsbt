# GitHub 上传指南

## 1. 初始化与上传

1. 解压本包，将根目录 `cliff-sbt-repro/` 作为仓库根目录。
2. 双盲投稿期间建议使用匿名 GitHub 组织/账号，并保留当前泛化的 `CITATION.cff`。
3. 本地执行：

```bash
git init
git add .
python reproduce/preflight_git_tracking.py --require-git
git status --short
git commit -m "Initial reproducibility release"
git branch -M main
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin main
```

`preflight_git_tracking.py` 会核对 `MANIFEST.sha256` 中的每一个证据文件是否真正被 Git 跟踪，避免“文件在 ZIP 中存在、却因 `.gitignore` 未上传 GitHub”的情况。

## 2. 推送前复现检查

```bash
make preflight-git
make verify
make test
make figures
make audit
```

也可以执行：

```bash
make all
```

## 3. 发布建议

- GitHub release 建议标记 `v1.0.1-manuscript-v7`。
- 随后在 Zenodo 归档，并将 DOI 写回 README、`CITATION.cff` 与论文 Code availability。
- 145 MB 完整证据包不要直接提交 Git；请放 Zenodo/OSF release，并更新 `evidence/full_evidence_manifest.json` 中的 `download_uri`。

## 4. 关于 CURE-OR 执行日志

仓库默认忽略普通 `*.log` 文件，但以下冻结证据日志被显式列入版本控制：

```text
experiments/cure_or/audit/PHASE1_EXECUTION.log
```

`.gitignore` 中已经加入例外规则。不要删除该文件或例外规则，否则严格 manifest 审计会失败。
