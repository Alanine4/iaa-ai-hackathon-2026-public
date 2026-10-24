# 办公室保费（office premium）公式草案与数据核对

用途：Part 2 报告中，对 20 张参考保单说明如何从预测的风险保费推到办公室保费。本文给出公式链草案和需要人来定的口径。数据取自 `data/iaa-ai-assisted-hackathon-2026-part-2/emerging/` 下的 `expenses.parquet`、`reference_policies.parquet`，由 `scripts/part2_expenses.py` 打印核对。

## 1. 官方原文（来源：`docs/references/part2_kaggle_data_page.md`）

- 文件清单表（第 44–45 行）：
  - "expenses | emerging | Cost item | Expense, commission, tax, reinsurance and profit assumptions for both markets"
  - "reference_policies | emerging | Policy | 20 policies for the office premium build-up in your report"
- expenses 小节（第 140–149 行）：列为 `item`（Cost item）、`developed`（Value in the developed market）、`emerging`（Value in the emerging market (blank if not applicable)）、`unit`（CU, share of premium, or share of property risk premium）。"Items: fixed expense per policy, variable expense, commission by channel, premium tax, catastrophe reinsurance and target profit margin."
- reference_policies 小节（第 151–153 行）："Twenty 2025 emerging-market policies taken from the test files (same columns, plus line). For each one, show in your written report how you build the office premium from your predicted risk premium using the expenses assumptions. This is assessed by the judging panel and does not affect the Kaggle score."
- 预测目标（第 19–23 行）："This is the risk premium, not the premium charged." "The prediction is a total for the row, not an annual rate. A row with exposure of 0.5 covers half a year."

`docs/references/part2_kaggle_overview_page.md` 中没有 office premium、expenses 或 reference policies 的相关文字。官方没有给出组合公式，下面的公式链是草案。

## 2. expenses.parquet 全表

8 行，4 列：`item`、`developed`、`emerging`、`unit`。本任务只用 `emerging` 列；`developed` 列列出仅供对照。

| item | developed | emerging | unit | 含义 | 量纲 | 按险种？ | 按渠道？ |
|---|---|---|---|---|---|---|---|
| fixed_expense_per_policy | 45.00 | 85.00 | CU | 每张保单的固定费用 | 金额（CU） | 全险种统一 | 否 |
| variable_expense | 0.12 | 0.15 | share of premium | 变动费用率 | 占保费比例 | 全险种统一 | 否 |
| commission_direct | 0.00 | 0.00 | share of premium | 直销渠道佣金率 | 占保费比例 | 全险种统一 | 仅 direct |
| commission_comparison_site | 0.10 | 空白（NaN） | share of premium | 比价网站渠道佣金率 | 占保费比例 | 全险种统一 | 仅 comparison_site |
| commission_broker | 0.15 | 0.20 | share of premium | 经纪人渠道佣金率 | 占保费比例 | 全险种统一 | 仅 broker |
| premium_tax | 0.12 | 0.08 | share of premium | 保费税率 | 占保费比例 | 全险种统一 | 否 |
| cat_reinsurance | 0.06 | 0.12 | share of property risk premium | 巨灾再保险成本 | 占财产险风险保费比例 | 仅财产险（见开放问题 3） | 否 |
| target_profit_margin | 0.05 | 0.08 | share of premium | 目标利润率 | 占保费比例 | 全险种统一 | 否 |

"按险种""按渠道"两列来自 item 名称和 unit，文件里没有险种列，也没有渠道列。

## 3. reference_policies.parquet 的 20 行

文件共 50 列，比测试文件多 `line`（险种）。险种分布：motor 7（row_id 以 M 开头）、buildings 7（B）、contents 6（C）。渠道：broker 17，direct 3，无 comparison_site。新业务（`new_business`）18 张，续保 2 张。以下只列与办公室保费推导相关的字段。`written_premium` 是现行定价下的已收保费（Part 1 数据页词汇表："Premium charged under the insurer's current pricing"，来源 `docs/references/kaggle_data_page.md` 第 122 行），不是本任务的风险保费。

| row_id | policy_id | line | channel | new_business | period_start | period_end | exposure | written_premium | discount_applied | voluntary_excess | compulsory_excess |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B00215707 | B75172079 | buildings | broker | True | 2025-01-06 | 2025-12-31 | 0.9863 | 272.44 | 0.00 | 0 | 350 |
| B00215809 | B88623113 | buildings | direct | True | 2025-01-10 | 2025-12-31 | 0.9753 | 237.22 | 0.05 | 0 | 350 |
| B00217584 | B13799940 | buildings | broker | True | 2025-01-13 | 2025-12-31 | 0.9671 | 323.36 | 0.00 | 0 | 350 |
| B00218282 | B74876592 | buildings | broker | True | 2025-01-11 | 2025-12-31 | 0.9726 | 230.80 | 0.00 | 250 | 350 |
| B00223369 | B73680946 | buildings | broker | True | 2025-01-03 | 2025-12-31 | 0.9945 | 204.56 | 0.00 | 0 | 350 |
| B00229821 | B92110463 | buildings | broker | True | 2025-01-04 | 2025-12-31 | 0.9918 | 285.60 | 0.00 | 250 | 350 |
| B00250679 | B05817658 | buildings | broker | True | 2025-11-22 | 2025-12-31 | 0.1096 | 180.72 | 0.00 | 0 | 350 |
| C00259568 | C16955632 | contents | broker | False | 2025-01-01 | 2025-12-31 | 1.0000 | 83.18 | 0.00 | 1000 | 350 |
| C00270331 | C97318122 | contents | broker | True | 2025-01-11 | 2025-12-31 | 0.9726 | 140.29 | 0.00 | 1000 | 350 |
| C00283470 | C11841593 | contents | broker | True | 2025-01-11 | 2025-12-31 | 0.9726 | 117.97 | 0.00 | 250 | 350 |
| C00286287 | C12450281 | contents | broker | True | 2025-01-13 | 2025-12-31 | 0.9671 | 269.23 | 0.00 | 0 | 350 |
| C00302985 | C36495409 | contents | broker | True | 2025-01-04 | 2025-12-31 | 0.9918 | 115.88 | 0.00 | 250 | 350 |
| C00314425 | C17515168 | contents | broker | True | 2025-01-13 | 2025-02-06 | 0.0685 | 102.87 | 0.00 | 250 | 350 |
| M00118569 | M75811658 | motor | direct | False | 2025-01-01 | 2025-12-09 | 0.9397 | 1095.48 | 0.05 | 0 | 250 |
| M00129413 | M41089337 | motor | broker | True | 2025-09-02 | 2025-12-31 | 0.3315 | 354.15 | 0.00 | 250 | 250 |
| M00153851 | M47069725 | motor | broker | True | 2025-01-15 | 2025-12-31 | 0.9616 | 414.72 | 0.00 | 0 | 250 |
| M00161473 | M92909747 | motor | broker | True | 2025-01-14 | 2025-12-31 | 0.9644 | 1131.78 | 0.00 | 250 | 250 |
| M00167496 | M61457154 | motor | broker | True | 2025-01-07 | 2025-12-31 | 0.9836 | 463.95 | 0.00 | 250 | 250 |
| M00176777 | M42023395 | motor | broker | True | 2025-01-04 | 2025-12-31 | 0.9918 | 443.61 | 0.00 | 0 | 250 |
| M00185311 | M83715550 | motor | direct | True | 2025-01-14 | 2025-12-31 | 0.9644 | 872.31 | 0.05 | 1000 | 250 |

exposure 小于 0.5 的行有 3 张：B00250679（0.1096）、C00314425（0.0685）、M00129413（0.3315），固定费用口径对这 3 张影响最大。

## 4. 公式链（草案）

符号：R = 该行预测的风险保费（行总额，不是年化，单位 CU）；exposure = 该行 `exposure`。以下每一步标明所用字段。

| 步骤 | 内容 | 公式 | 使用的字段 |
|---|---|---|---|
| 1 | 风险保费 | R = 最终提交文件中该 row_id 的 prediction | 来自 `work/` 中最终提交文件，不是 expenses 字段 |
| 2 | 巨灾再保险成本（仅 buildings、contents；motor 取 0） | Cat = 0.12 × R | `cat_reinsurance` |
| 3 | 固定费用（每张保单） | F = 85 CU；是否乘 exposure 见开放问题 1 | `fixed_expense_per_policy` |
| 4 | 渠道佣金率 c | broker 取 0.20；direct 取 0.00；comparison_site 为空白（见开放问题 2） | `commission_broker`、`commission_direct`、`commission_comparison_site` |
| 5 | 与办公室保费成比例的各项合计 k | k = 0.15 + c + 0.08 + 0.08，即 variable_expense + c + premium_tax + target_profit_margin | `variable_expense`、`premium_tax`、`target_profit_margin`，以及第 4 步的 c |
| 6 | 办公室保费 | P = (R + Cat + F) ÷ (1 − k) | 第 1–5 步 |

分母（只做加减，数值来自上表）：broker 的 1 − k = 1 − 0.15 − 0.20 − 0.08 − 0.08 = 0.49；direct 的 1 − k = 1 − 0.15 − 0.00 − 0.08 − 0.08 = 0.69。

各费用项在 P 中对应的金额，可在报告表里分列：

- 变动费用 = variable_expense × P
- 佣金 = c × P
- 保费税 = premium_tax × P
- 利润 = target_profit_margin × P
- 巨灾再保 = Cat（第 2 步，仅财产险）
- 固定费用 = F（第 3 步）

验算：风险保费 + 巨灾再保 + 固定费用 + 变动费用 + 佣金 + 保费税 + 利润 = P。

公式链基于 unit 列的字面含义：`share of premium` 的各项按办公室保费 P 取比例，故放在分母；`share of property risk premium` 的 `cat_reinsurance` 按风险保费 R 取比例，故是加在 R 上的成本；`fixed_expense_per_policy` 单位是 CU，故是加在分子上的金额。官方没有写组合方式，这个结构是对 unit 的解读，需要队员确认。

## 5. 必须由人决定的开放问题

| # | 问题 | AI 建议 | 理由 |
|---|---|---|---|
| 1 | 固定费用 85 CU 是"每张保单"，参考保单里有 exposure 只有 0.0685、0.1096、0.3315 的行，怎样分摊？ | 按 exposure 折算：F = 85 × exposure；报告里写明口径，并在脚注给出不折算时的结果作对照 | 风险保费 R 是行总额（官方：exposure 0.5 的行覆盖半年），R 与 F 同在一个分子里，时间口径应一致；同一保单一年内有多行时（中途变更），不折算会重复收取。反方理由：短期保单的获客和开户成本并不随时间缩短，不折算更贴近真实成本 |
| 2 | `commission_comparison_site` 在 emerging 为空白，当 0 处理是否合理？ | 当 0 处理，并在报告中写明；同时注明 emerging 数据里比价网站渠道约占 5%（`docs/PART2_INTAKE.md` N2），这条假设对 20 张参考保单无影响（参考保单没有该渠道） | 官方对该列的说明是 "blank if not applicable"，按字面读就是不适用，当 0 不违背官方；但数据里存在该渠道，当 0 会低估成本。若要审慎，可在敏感性说明中给出改用 developed 的 0.10 时的差异 |
| 3 | 巨灾再保险成本只加房屋/家财，还是也加车险？ | 只加 buildings 和 contents，motor 取 0 | unit 写的是 "share of property risk premium"，motor 不是财产险。"property" 是否包含 contents，官方没有明说，建议两个险种都加，并在报告里写明 |
| 4 | 保费税 `premium_tax`（0.08）/IPT 在哪一步加？ | 放在分母，和变动费用、佣金、利润一起，按办公室保费取比例 | unit 是 "share of premium"。另一种读法是先算不含税保费，再乘 (1 + 税率)；两种读法税基不同、结果不同，需要队员选定并在报告里写出公式。保费税是否也对固定费用征收，官方没有写明 |
| 5 | 对 20 张参考保单要不要做保费上下限（cap/floor）？ | 主表不做上下限；另加一列 `written_premium` 作对照，对偏离大的行在备注里解释 | 上下限属于商业决策，官方没有提供上下限参数；加上限会遮住模型预测本身。`written_premium` 是现行定价的结果，只作参考 |
| 6 | `discount_applied`（0.05，参考保单中 3 张）是否要在办公室保费里扣除？ | 不扣 | 办公室保费从风险保费推出，不依赖现行定价；折扣属于 `written_premium` 的构成，只在对照列里体现 |
| 7 | 风险保费预测为 0 时怎么办？ | 照公式计算，P = F ÷ (1 − k)（财产险同理，Cat 为 0） | 提交要求预测非负（官方："predictions zero or above"），公式在 R = 0 时仍有定义；固定费用不随 R 消失 |
| 8 | 金额保留几位小数？ | 中间步骤不取整，最终展示保留两位小数 | 避免舍入误差累计，`written_premium` 本身是两位小数 |

## 6. 交付物样式

报告里放一张 20 行表，每行一张参考保单，按 line、row_id 排序。

| 列 | 来源 | 状态 |
|---|---|---|
| row_id | reference_policies | 已有 |
| line（险种） | reference_policies | 已有 |
| channel（渠道） | reference_policies | 已有 |
| exposure | reference_policies | 已有 |
| 风险保费 R | `work/` 中最终提交文件里该 row_id 的 prediction | 待填（现在留空，不编数字） |
| 巨灾再保（仅财产险） | 0.12 × R | 待算 |
| 固定费用 | 85（或 85 × exposure，取决于开放问题 1） | 待算 |
| 变动费用 | 0.15 × P | 待算 |
| 佣金 | c × P（broker 0.20，direct 0.00） | 待算 |
| 保费税 | 0.08 × P | 待算 |
| 利润 | 0.08 × P | 待算 |
| 办公室保费 P | (R + 巨灾再保 + 固定费用) ÷ (1 − k) | 待算 |
| written_premium（对照） | reference_policies | 已有 |

表下注明：币种为 CU；所用口径（开放问题 1、2、3、4 的最终选择）；风险保费来源文件和版本。

## 7. 核对结果

- expenses：8 行 4 列，字段名、数值、单位与 parquet 一致（由 `scripts/part2_expenses.py` 打印）。emerging 列中 `commission_comparison_site` 为 NaN，其余 7 项有值。
- reference_policies：20 行，row_id 全部列出，险种分布 motor 7 / buildings 7 / contents 6，渠道 broker 17 / direct 3，与 `docs/PART2_INTAKE.md` 第 6 节的描述一致。
- 本文没有填入任何风险保费或办公室保费数值。
