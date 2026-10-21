# 待办清单

规则：每做完一项就把 `[ ]` 改成 `[x]`，并在行尾写上完成日期和产出文件。新发现的事项加到对应阶段的末尾。
Part 1 截止：约 2026-10-23（以 Kaggle 页面倒计时为准，UTC）。

## 阶段 0：准备
- [x] 下载 11 个数据文件到 `data/` — 2026-10-21
- [x] 建立 Python 环境 `.venv`（pandas、pyarrow、lightgbm、sklearn、statsmodels、matplotlib） — 2026-10-21
- [x] 存档 Overview、Data、Rules 三页原文到 `docs/references/` — 2026-10-21
- [x] 逐个体检 11 个文件 → `scripts/profile_data.py`、`docs/data_profile.txt` — 2026-10-21
- [x] 核实数据质量问题 → `scripts/check_quality.py`、`docs/data_quality_checks.txt` — 2026-10-21
- [x] 写数据说明与比赛说明 → `docs/DATA_GUIDE.md`、`docs/COMPETITION.md` — 2026-10-21
- [x] 初始化 git 仓库，配置 `.gitignore`（排除 `data/`、`.venv/`、`output/`），做第一次提交 — 2026-10-21
- [x] 建 `docs/AI_LOG.md` 和修改日志 `docs/CHANGELOG.md` — 2026-10-21
- [x] 推送到 GitHub 私有仓库 https://github.com/Alanine4/iaa-ai-hackathon-2026（不含 data/、output/） — 2026-10-21
- [ ] 邀请队友为 collaborator（要先拿到队友的 GitHub 用户名；队友也必须在 Kaggle 上合入同一个队伍）

## 阶段 1：目标变量（第 1 天）
- [x] 清洗：exposure 重算（C001）、customer_id 补零（C002）、construction_type 统一（C004） — 2026-10-21，`src/aiiaa/clean.py`
- [x] 确认 construction_type 里 W 和 T 的含义：同一类，T 用于 2021–22 年，W 用于 2023–24 年，都映射为 timber（C004） — 2026-10-21
- [x] 确定 sum_insured 异常值规则：大于 500 万的除以 1000，并加标记列（C003） — 2026-10-21
- [x] 理赔对齐到保单行，去掉拒赔，计算 incurred − recoveries，对不上的单独列出（C005） — 2026-10-21，`src/aiiaa/target.py`
- [x] 发展调整：按险种用按金额加权的报案延迟曲线（C006） — 2026-10-21，`src/aiiaa/development.py`
- [x] 自动化测试 18 个全部通过；全量合计核对差额为 0 — 2026-10-21，`tests/`、`docs/target_summary.txt`
- [ ] 人工复核：C003 的阈值、C004 的映射、房屋 2024 年发展后水平（481，按趋势约 458）是否偏高
- [ ] 按险种细分理赔原因的报案曲线（车险人伤报案慢），看是否改善 2024 年的估计

## 阶段 2：特征与验证（第 1 天）
- [ ] 合并 customers、addresses、weather（按年份）
- [ ] 特征工程（年龄、房龄、历史理赔经验等）；删掉 renewal_decision（Q9）
- [ ] 验证方案：2021–2023 训练、2024 验证，用 Tweedie D²（power 1.5）评分
- [ ] 基准模型：每个险种一个 Tweedie GLM
- [ ] LightGBM Tweedie 模型（以 exposure 作为 offset 或权重）
- [ ] 第一次提交 Kaggle，确认格式正确、拿到分数

## 阶段 3：改进与诊断（第 2 天）
- [ ] 调参；比较 GLM 与 LightGBM，考虑两者融合
- [ ] 诊断：实际 vs 预测（按分位）、lift 曲线、各特征的影响（SHAP / 部分依赖）
- [ ] 检查训练/测试分布差异（Q10），写入报告
- [ ] 用 2021–2024 全量数据重新训练，生成最终提交
- [ ] 在 Kaggle 上选定 Final Submission

## 阶段 4：为 Part 2 做准备
- [ ] 把流程整理成一条命令就能重跑的脚本（换数据路径就能跑）
- [ ] 写 README：环境、运行步骤、目录结构
- [ ] 报告草稿：方法、决策、AI 使用、假设、局限、管控措施
- [ ] Part 2 数据发布后：对比新旧市场差异，再拆新的待办
