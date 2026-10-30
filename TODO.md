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
- [x] 初始化 git 仓库，配置 `.gitignore`（排除 `.venv/`、`output/`；data/ 自 C007 起纳入版本管理），做第一次提交 — 2026-10-21
- [x] 建 `docs/AI_LOG.md` 和修改日志 `docs/CHANGELOG.md` — 2026-10-21
- [x] 推送到 GitHub 私有仓库 https://github.com/Alanine4/iaa-ai-hackathon-2026（不含 output/） — 2026-10-21
- [x] 原始数据作为只读备份提交到私有仓库，附 SHA256 清单和完整性测试（C007） — 2026-10-21
- [x] 邀请队友为 collaborator：GitHub 用户 OBLEMOS 已有写权限（gh api 核对）；Kaggle 已同队 — 2026-10-26

## 阶段 1：目标变量（第 1 天）
- [x] 清洗：exposure 重算（C001）、customer_id 补零（C002）、construction_type 统一（C004） — 2026-10-21，`src/aiiaa/clean.py`
- [x] 确认 construction_type 里 W 和 T 的含义：同一类，T 用于 2021–22 年，W 用于 2023–24 年，都映射为 timber（C004） — 2026-10-21
- [x] 确定 sum_insured 异常值规则：大于 500 万的除以 1000，并加标记列（C003） — 2026-10-21
- [x] 理赔对齐到保单行，去掉拒赔，计算 incurred − recoveries，对不上的单独列出（C005） — 2026-10-21，`src/aiiaa/target.py`
- [x] 发展调整：按险种用按金额加权的报案延迟曲线（C006） — 2026-10-21，`src/aiiaa/development.py`
- [x] 自动化测试 18 个全部通过；全量合计核对差额为 0 — 2026-10-21，`tests/`、`docs/target_summary.txt`
- [ ] 人工复核：C003 的阈值、C004 的映射、房屋 2024 年发展后水平（481，按趋势约 458）是否偏高
- [x] 按险种细分理赔原因的报案曲线（车险人伤报案慢），看是否改善 2024 年的估计 — 2026-10-21，C022，数值无变化（按成本占比混合后与合并曲线完全一致），保留为构件，`docs/CHANGELOG.md`

## 阶段 2：特征与验证（第 1 天）
- [x] 合并 customers、addresses、weather（按年份） — 2026-10-21，`src/aiiaa/features.py`
- [x] 特征工程（客户年龄、入保年数、房龄、同一保单往年理赔经验）；删掉 renewal_decision（Q9）和 customer_service_contacts（C008） — 2026-10-21
- [x] 全字段泄漏扫描 → `docs/scan_leakage.txt` — 2026-10-21
- [x] 验证方案：2021–2023 训练、2024 验证，Tweedie D²（power 1.5），趋势调整（C009） — 2026-10-21，`src/aiiaa/evaluate.py`
- [x] 模型代码：GLM、GAM（C011）、LightGBM、XGBoost，51 个测试全部通过 — 2026-10-21，`src/aiiaa/models.py`
- [x] 跑基准：GLM / GAM / LightGBM / XGBoost 验证 + 全量重训 + 5 个提交文件（含平均融合） — 2026-10-21，`docs/validation_results.md`、`output/submissions/`
- [ ] 查验证集上整体偏高 7–11% 的原因（不含趋势时偏差较小，疑似 3 年数据外推的趋势系数偏高；房屋 2025 年趋势系数 1.199 需要复核）
- [ ] 设置 Kaggle API 凭据（需要你本人操作），之后可以用 CLI 提交
- [x] 模型选择：按险种比较验证分数，决定用单个模型还是融合，写入 CHANGELOG — 2026-10-21，C021 按险种融合权重（要求两折都不劣于等权才采用），`docs/CHANGELOG.md`
- [x] 第一次提交 Kaggle：融合模型，公开榜 0.12453，第 19 名 — 2026-10-21，`docs/SUBMISSIONS.md`
- [ ] 上传两个趋势对照版本（`work/sub_blend_trend000.zip`、`work/sub_blend_trend050.zip`），把分数告诉我，用来决定趋势用多少（C016）

- [x] 数据全览文档：11 个文件逐字段的中文含义、例子、类型、来源（官方 / 推断）和处理方式 — 2026-10-21，`docs/DATA_WALKTHROUGH.md`
- [x] 变量类型修正：branch_id 等"数字形式的编号"在 GLM/GAM 里应当按类别处理，不能按连续数字处理 — 2026-10-22，C026 已直接删除 branch_id（无频率信号），`docs/CHANGELOG.md`、`docs/AUDIT_2026-10-22.md`

## 阶段 3：改进与诊断（第 2 天）
- [x] 调参；比较 GLM 与 LightGBM，考虑两者融合 — 2026-10-21，C017/C018，`docs/CHANGELOG.md`
- [ ] 诊断：实际 vs 预测（按分位）、lift 曲线、各特征的影响（SHAP / 部分依赖）（部分完成 2026-10-22：独立审计查了各列频率离散度与树模型增益占比，`docs/AUDIT_2026-10-22.md`；分位对比、lift、SHAP 未做）
- [ ] 检查训练/测试分布差异（Q10），写入报告（部分完成 2026-10-22：审计确认 Q10 漂移内部一致，renewal_count 每年 +1.0，`docs/AUDIT_2026-10-22.md`；报告未写）
- [ ] 用 2021–2024 全量数据重新训练，生成最终提交
- [ ] 在 Kaggle 上选定 Final Submission
- [ ] 人工复核 C026 捆绑删除的三列（document_preference、marital_status、occupation 是一起测试的，没有逐列单独验证）
- [ ] 分峰模型用新特征重跑（可选，时间允许时）
- [ ] 最终提交前在 Kaggle 选定 Final Submission

## 阶段 4：为 Part 2 做准备
- [x] 把流程整理成一条命令就能重跑的脚本（换数据路径就能跑）（`reproduce.ps1`，README「Reproducing」两节；核对 2026-10-30）
- [x] 写 README：环境、运行步骤、目录结构（`README.md`，核对 2026-10-30）
- [x] 报告草稿：方法、决策、AI 使用、假设、局限、管控措施（2026-10-30，`docs/report/report.tex`，按四条评分标准排，评分部分 5 页 + 附录；`docs/REPORT_DRAFT.md` 为素材稿）
- [ ] Part 2 数据发布后：对比新旧市场差异，再拆新的待办

## 阶段 5：Part 2 新兴市场
- [x] 按市场切换数据路径和输出目录（`--market` / `AIIAA_MARKET`，默认仍是 Part 1） — 2026-10-24，C034，`src/aiiaa/markets.py`
- [x] 两个市场统一字段：年龄改成年龄段、性别、职业清洗（拆出自雇标记）、构造类型新代码 — 2026-10-24，C035，`src/aiiaa/features.py`
- [x] 新兴市场目标变量：借用发达市场的报案曲线，核对差额为 0，并对比两边报案曲线 — 2026-10-24，C037，`docs/part2_emerging_target_summary.txt`
- [x] 新兴市场验证（2023→2024 时间折 + 按保单分组 5 折）和五个模型的融合权重 — 2026-10-24，C036，`output/emerging/blend_weights_<line>.json`
- [x] 只用新兴市场数据的提交文件（未上传） — 2026-10-24，C036，`work/p2_emerging_only.csv`、`work/p2_emerging_only.zip`
- [x] 合并两个市场训练（发达市场权重 0.3）跑三个险种并对比：家财 0.1072 vs 0.0779、房屋 0.1065 vs 0.0410、车险 0.0915 vs 0.0762（时间折 d2_cal），分组均值也全赢，三条线全部采用；GAM 因内存超 4 GB 未进 pooled — 2026-10-25，`output/emerging/pooled_w0.3/`、`work/p2_pooled_w03_half.zip`（P2-S04）
- [x] 合并权重探针（家财）：权重 1.0 对 0.3，时间折 0.1069 vs 0.1072、分组均值 0.1128 vs 0.1100，平均 +0.0013 < 0.002，0.3 保留；树模型单独受益（LGBM +0.0067、CAT +0.0145） — 2026-10-25，`output/emerging/pooled_w1/blend_weights_contents.json`
- [x] C039 调参（pooled 家财，LGBM 13 配置 + XGB 10 配置 × 6 折）：没有配置在时间折和分组折上同时赢默认值（LGBM 叶子 15：分组 +0.0030 / 时间折 −0.0063；XGB 默认最好），参数不变；"叶子 3000 行太粗"的假设在 pooled 上不成立 — 2026-10-25，`docs/cv_emerging_trees_pooled_w0.3.md`
- [x] 房屋线权重 1.0 探针：时间折 0.1169 vs 0.1065、分组均值 0.1180 vs 0.1132，平均 +0.0076，房屋采用 1.0（家财保留 0.3） — 2026-10-25，`output/emerging/pooled_w1/blend_weights_buildings.json`，P2-S05
- [x] 车险线权重 1.0 探针：时间折 0.0908 vs 0.0915、分组均值 0.0730 vs 0.0740，略差，车险保留 0.3；三线权重 {车险 0.3，房屋 1.0，家财 0.3} 写进 `markets.py`，blend.py 支持 `--developed-weight per-line`（C040），仓库命令重现 P2-S05 逐字节一致 — 2026-10-25
- [x] 优化探针 C041（跨运行融合）与 C042（GLM 市场交互项）：都未达采纳标准，C042 的形式达标版因新业务水平未验证被否决；新业务分段核查证据噪声大，不做调整 — 2026-10-25，`docs/CHANGELOG.md` C041/C042、`docs/part2_new_business_check.md`、AI_LOG A23
- [x] 水平校准（C043）：6 个标量探针（P2-S06–S11）；车险水平偏低，改用新兴市场自身完整趋势 1.1606（`mixed-full`），房屋家财不变；P2-S10 0.15414 — 2026-10-25，`docs/SUBMISSIONS.md`
- [x] 审计对策（C044/C045）：测试比例加权视图成为门槛；新业务加权训练、只改早停、车险去 GLM 三项都否决；家财双运行融合在新门槛下采用 → P2-S12 0.15474 — 2026-10-26，`docs/AUDIT_PART2.md`、CHANGELOG C044/C045
- [x] 新模型实验 C046（新兴市场树模型调参→融合候选，三线否决）、C047（新业务专用模型 + 分段融合：车险新业务段采纳，房屋续保段按设计否决）→ P2-S13 0.15577 — 2026-10-26，CHANGELOG C046/C047、`docs/part2_validation/`
- [ ] Kaggle 最终选择：候选 P2-S15（0.15807，种子平均版，事先规则已通过，2026-10-30，`docs/SUBMISSIONS.md`）；对冲 P2-S14（0.15781）
- [x] Part 1 报告和视频把截止后融合（0.7 S09 + 0.3 OBLEMOS，迟交、不计排名，私榜 0.13718）与计入排名的 S09（私榜 0.13689，第 12 / 51）并列介绍；草稿已写入 `docs/REPORT_DRAFT.md`、`docs/VIDEO_SCRIPT.md`，定稿时保留（2026-10-30：报告表 3 和附录 B、仪表盘总览都已并列展示，标注 late/not ranked）
- [ ] 把 `docs/report/report.tex` 贴进 Overleaf 编译，确认第 4 节结束在第 5 页、附录从第 6 页起（队长）
- [ ] 队长签字 `docs/HUMAN_REVIEW.md` R12–R16
- [ ] 用 `docs/dashboard/index.html`（或 artifact 链接）录 5 分钟视频，台词 `docs/VIDEO_SCRIPT.md`（664 词，约 4:55，建议第 5 段删约 10 词）
- [x] 发达市场默认路径复验：build_target 三条线 target 表与 `output/target_*.parquet` 完全一致、报告与 `docs/target_summary.txt` 一致；S09 融合用当前 blend.py 重现，与 `work/sub_v3_seeds.csv` SHA256 相同 — 2026-10-25，`output/emerging/check_dev_target.log`
- [x] 半趋势系数重算：车险 1.0175、房屋 1.0948、家财 1.0636，与 `markets.py` 一致 — 2026-10-25，`output/emerging/check_dev_target.log`
- [ ] GLM/GAM 在新兴市场严重过拟合（房屋时间折 D² 为负，单行预测 40 万），考虑加大正则或只用树模型
- [x] 上传 `work/p2_emerging_only.zip` 到 Kaggle 并记录分数 — 2026-10-24，P2-S01 公开榜 0.14370（第 8 / 10），`docs/SUBMISSIONS.md`
- [x] 趋势探针 P2-S02（developed-half）：0.14460，比无趋势高 0.0009，Part 2 默认改用 developed-half — 2026-10-24，`docs/SUBMISSIONS.md`
- [x] 办公室保费公式草案（expenses 8 项、20 张参考保单全表、公式链、8 个开放问题） — 2026-10-24，`docs/OFFICE_PREMIUM_DRAFT.md`、`scripts/part2_expenses.py`；**开放问题待你和队友定**
- [x] 人工复核签字单：11 项 AI 关键决策 + 7 条被纠正的 AI 输出，每项附文件:行号与数字 — 2026-10-24，`docs/HUMAN_REVIEW.md`；**等队长逐项签字**，签完抄进 AI_LOG
- [ ] 修正过时文字：`docs/DATA_GUIDE.md` 第 107–108 行 Q3/Q4 仍写"待定/未确认"，应改为 C003/C004 的结论（worker 顺手发现）
- [x] 报告第 7 节（新市场适配：差异、流程改动、验证设计、结果表、办公室保费公式、局限），约 590 词 — 2026-10-25，`docs/REPORT_DRAFT.md`
- [x] 报告压到 5 页以内并加第 8 节（2026-10-30，`docs/report/report.tex` 第 1.4 节，保费表用 P2-S15 预测，C060）：20 张参考保单的办公室保费表（等最终预测文件和队友定口径）
- [x] 视频脚本（5 分钟，约 700 字）（2026-10-29，`docs/VIDEO_SCRIPT.md`，测试数已改为 340）
- [x] 邀请队友为 GitHub collaborator（OBLEMOS，写权限） — 2026-10-26
- [x] 更正：队友 Part 1 文件的 0.1356 是私榜分，公开榜最高 0.12758，都低于 S09；S09 选择正确。队友 10/27–28 在 Part 2 交了 6 个基于 S12/S13 的变体（最高 0.15563，未超过 S13），记在 SUBMISSIONS T01–T05 — 2026-10-29
