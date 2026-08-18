# Cliff SBT 轻量级复现仓库

本仓库对应论文 v7，目标是**直接上传 GitHub、快速核验、便于社区复用**。它不是 145 MB 完整证据档案的简单压缩，而是经过清理后的三级复现结构：

1. **无外部数据的快速核验**：从 compact committed evidence 核验论文核心数字，并重建全部主图和 Extended Data 图；
2. **方法复用**：内置 `sbt-monitor` 源码，可在用户自己的身份配对部署流上计算 transport ledger；
3. **正式实验重跑**：保留四个实验域的正式代码、配置和协议；大型公开数据、特征张量、checkpoint 与缓存按说明另行获取。

## 一键开始

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-lite.txt
python -m pip install -e tooling/sbt-monitor[warning]

python reproduce/preflight_git_tracking.py
python reproduce/verify_compact_evidence.py
python reproduce/make_figures.py --evidence-dir evidence/compact --out-dir figures/rebuilt
python -m pytest tooling/sbt-monitor/tests -q
python reproduce/audit_repository.py --strict
# 也可以：make all
```

## 清理原则

已删除：大型原始图像、35 MB 特征张量、模型 checkpoint、sklearn 缓存、重复图件、历史探索输出和可由 compact 表重建的中间数组。

已保留：论文/SI、正式图、sbt-monitor、正式实验代码与冻结协议、关键行级/汇总证据、失败与 STOP 的文档边界、完整档案哈希桥接文件。

## 科学边界

- Task SBT 需要真实 outcome 与身份配对；
- operational boundary 由用户声明，工具不提供“安全阈值”；
- outcome-blind prediction-state transport 只是代理，不等于真实 task SBT；
- warning 必须在用户自己的场域中校准，不发布万能预训练报警器。

双盲阶段的作者信息保持泛化；实名公开前请更新 `CITATION.cff` 和永久 DOI。
