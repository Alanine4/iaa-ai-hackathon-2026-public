# Video script: IAA AI Assisted Hackathon 2026

Length 5:00. Speech runs to 4:50 and the last 10 seconds are a buffer. Pace is about 135 words per minute. Word counts count the spoken column only (left column).

## Segment 1, 0:00-0:30: who we are and the task (78 words)

On screen: title card, then `README.md` section "What the competition asks".

| Spoken (English) | 中文对照 |
|---|---|
| Hello, we are team Yichuan Lin: Yichuan Lin and Oussama El Hadi, and this is our entry to the IAA AI Assisted Hackathon 2026. | 大家好，我们是 Yichuan Lin 队，成员林逸川和 Oussama El Hadi，这是我们在 IAA AI Assisted Hackathon 2026 的参赛作品。 |
| The task, in one sentence: predict the final claim cost of every 2025 policy row, for motor, home buildings and home contents. | 任务一句话讲完：预测 2025 年每一条保单行最终的理赔成本，涉及车险、房屋结构险和家财险。 |
| There are two markets. Part 1 is a developed market with four years of history, and Part 2 is a smaller emerging market. | 一共有两个市场。第一部分是有四年历史数据的成熟市场，第二部分是规模小得多的新兴市场。 |
| Entries are scored with Tweedie D squared, power 1.5. | 评分指标是 Tweedie D²，power 取 1.5。 |

## Segment 2, 0:30-1:30: approach (136 words)

On screen: `docs/REPORT_DRAFT.md` sections 2 and 3 (target construction, model list), ending on the adoption-rule sentence.

| Spoken (English) | 中文对照 |
|---|---|
| First, the target. We match each claim to the policy row that contains its loss date, and we subtract recoveries. | 先说目标变量。每笔理赔按出险日期对到它所属的保单行，再扣掉追偿金额。 |
| Recent claims are still being reported, so we develop them with a reporting-lag curve for each line. | 近期的理赔还没报完，所以我们按险种用报案延迟曲线，把它们发展到最终水平。 |
| For motor bodily injury, only 14 percent of the cost is reported after 30 days. | 比如车险的人身伤害，出险 30 天后只报了 14% 的成本。 |
| Each claim is capped at the 99.5th percentile, and a load adds the capped excess back. | 每笔理赔在 99.5 分位处封顶，再用一个附加系数把被封掉的部分补回来。 |
| Then we fit five Tweedie models: a GLM, a GAM, LightGBM, XGBoost and CatBoost, and blend them with weights chosen per line. | 然后拟合五个 Tweedie 模型：GLM、GAM、LightGBM、XGBoost 和 CatBoost，再按险种选权重做融合。 |
| Validation is by time, with two expanding folds, never at random. | 验证按时间来，用两个逐年扩展的折，不做随机划分。 |
| A change is adopted only if the mean gain is at least 0.002 and it loses on neither fold. | 一项改动只有在平均提升至少 0.002、并且两个折都不退步时，才会被采纳。 |
| We fixed that rule before running the experiments, and the blend weight check is now code. | 这条规则是在做实验之前就定好的，融合权重的检查现在也写进了代码。 |

## Segment 3, 1:30-2:30: AI-assisted workflow (127 words)

On screen: `docs/AI_LOG.md` (rows A09 and A18), a `docs/CHANGELOG.md` row with its code references, and the sign-off note in `docs/HUMAN_REVIEW.md`.

| Spoken (English) | 中文对照 |
|---|---|
| We used Claude Code in three roles. | 我们让 Claude Code 分三种角色工作。 |
| A coordinator plans and decides. Workers implement, and write the tests first. An independent auditor only reads and reports. | 协调者负责规划和决策；工人负责实现，并且先写测试；独立审计员只读代码和数据，只出报告。 |
| The suite now has 340 tests. The raw data is never edited, and every fix lives in code. | 现在测试一共有 340 个。原始数据从不修改，所有修正都写在代码里。 |
| Every change has a CHANGELOG number, and the code comments cite it. | 每一处改动都有一个 CHANGELOG 编号，代码注释里也会引用这个编号。 |
| Every AI step is logged, with how we checked it. Our team lead signed off eleven key decisions on 25 October. | 每一步 AI 工作都记在日志里，并写明我们怎么核对的。队长在 10 月 25 日对 11 项关键决策签了字。 |
| We also reject AI output. The coordinator designed per-peril reporting curves, C022. The worker implementing them showed they change nothing. | 我们也会否决 AI 的输出。协调者设计了按出险原因分开的报案曲线，也就是 C022。负责实现的工人用数值证明，它不会改变任何结果。 |
| Four other ideas, C023, C024, C031 and C032, were rejected on validation. | 另外四个想法，C023、C024、C031 和 C032，在验证中被否决。 |
| The audits found real issues: B1 to B8 in Part 1, and P2-F1 to F10 in Part 2. | 审计确实查出了问题：第一部分有 B1 到 B8，第二部分有 P2-F1 到 F10。 |

## Segment 4, 2:30-3:45: what changed for the new market (165 words)

On screen: `docs/REPORT_DRAFT.md` section 7.1 and the section 7.4 table (emerging only against pooled), then the P2-F1 row of `docs/AUDIT_PART2.md`, then `docs/SUBMISSIONS.md` rows P2-S10 and P2-S13.

| Spoken (English) | 中文对照 |
|---|---|
| Then the second market arrived. It has about one tenth of the data, covering only two years. | 然后第二个市场的数据发下来了。数据量大约只有原来的十分之一，只覆盖两年。 |
| 76 percent of its test rows are new business, against 15 percent before. | 它的测试行里有 76% 是新业务，之前只有 15%。 |
| Motor frequency is 34 percent higher, and severity is 42 percent lower. | 车险的出险频率高了 34%，平均赔款金额低了 42%。 |
| So each market became a configuration, with one shared schema, and we borrowed the developed reporting curves, because two years give no tail. | 所以我们把每个市场做成一套配置，字段统一成同一个结构；又因为两年的数据看不到长尾，就借用了成熟市场的报案曲线。 |
| Pooling the developed rows into training, at a weight of 0.3, lifted contents from 0.078 to 0.107 on the time fold. | 把成熟市场的数据按 0.3 的权重并进来一起训练后，家财险在时间折上的分数从 0.078 提高到 0.107。 |
| Then our audit found that validation was 92 percent renewals, while the test is mostly new business. We made a test-mix check part of the gate. Weighted to the test mix, validation matched the public board. | 接着审计发现，验证集有 92% 是续保，而测试集大部分是新业务。于是我们把按测试集比例加权的检查，加进了采纳门槛。这样加权之后，验证结果和公榜的变化方向一致。 |
| Motor level looked low, so we moved to the emerging market's own trend, 1.16. For motor we also added a separate new-business blend, which passed all six validation views. | 车险的整体水平偏低，所以我们改用新兴市场自己的趋势系数 1.16。另外我们给车险的新业务单独做了一套融合，它在六个验证视角上全部通过。 |
| We rejected interaction terms, new-business weighted training, and re-tuning, because none passed the gate. | 市场交互项、按新业务加权的训练、重新调参，这几项都没有通过门槛，所以被我们否决了。 |

## Segment 5, 3:45-4:35: results (119 words)

On screen: the Kaggle leaderboards for Part 1 and Part 2, the Part 1 post-deadline block of `docs/SUBMISSIONS.md`, then the report's office premium formula (`docs/REPORT_DRAFT.md` section 7.5) and the build-up table for the 20 reference policies.

| Spoken (English) | 中文对照 |
|---|---|
| In Part 1, our ranked final scored 0.13689 on the private board, rank 12 of 51. The blend we present, a late submission and not ranked, scored 0.13718. | 第一部分，我们计入排名的最终提交在私榜得到 0.13689，名列 51 支队伍中的第 12。我们在报告里介绍的融合版本是截止后的迟交提交，不计排名，得分 0.13718。 |
| In Part 2, the public score rose from 0.14370 with emerging data only, to 0.15193 with pooled training, and now 0.15807 with seed-averaged trees. At recording time we rank 18th of 38 teams, and the leader has 0.176. | 第二部分，公榜分数从只用新兴市场数据的 0.14370，升到合并训练后的 0.15193，现在用种子平均的树模型达到 0.15807。录制时我们排在 38 支队伍中的第 18，榜首是 0.176。 |
| For the 20 reference policies, office premium is risk premium, plus catastrophe reinsurance for property and a fixed expense of 85, divided by one minus the loadings for expense, commission, tax and profit. | 对那 20 张参考保单，办公室保费等于风险保费，加上财产险的巨灾再保险成本和 85 的固定费用，再除以 1 减去变动费用、佣金、税和利润这几项的比例之和。 |
| Our reporting curves are borrowed and the new-business evidence is thin. Part of the motor level came from leaderboard probes. | 局限有两点：报案曲线是借来的，新业务的证据很少；车险的整体水平有一部分来自对公榜的试探提交。 |

## Segment 6, 4:35-4:50: closing (39 words)

On screen: the repository file tree: `README.md`, `reproduce.ps1`, `tests/`, `docs/CHANGELOG.md`.

| Spoken (English) | 中文对照 |
|---|---|
| Code, tests and the change log are in the repository. | 代码、测试和修改日志都在代码仓库里。 |
| One script rebuilds Part 1 from the raw data, and a market setting runs the emerging data. | 一个脚本就能从原始数据重建第一部分的结果，换一个市场设置，同一套流程也能跑新兴市场的数据。 |
| Next, we would test a developed-model offset, an idea from our audit. | 下一步，我们想试一试把成熟市场模型的预测当作偏移量，这是审计提出来的思路。 |

Total spoken words: 664 (about 4:55 at 135 words per minute, 5 seconds over the 4:50 target; trim about 10 words, segment 5 is the candidate).
