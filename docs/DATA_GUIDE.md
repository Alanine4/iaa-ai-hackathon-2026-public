# 数据说明（Part 1，发达市场）

依据：`docs/references/kaggle_data_page.md`（官方字段说明）、`docs/data_profile.txt`（`scripts/profile_data.py` 的输出）、`docs/data_quality_checks.txt`（`scripts/check_quality.py` 的输出）。数字均来自 2026-10-21 实际跑出的结果。

数据位置：`data/iaa-ai-assisted-hackathon-2026-part-1/`，11 个 parquet 文件，共约 198 MB。

## 1. 怎么打开

parquet 是按列压缩的表格文件，Excel 打不开，要用 Python：

```python
import pandas as pd
d = r"C:\Projects\AIIAA\data\iaa-ai-assisted-hackathon-2026-part-1"
m = pd.read_parquet(d + r"\policies_motor.parquet")
m.shape; m.head().T; m.info(); m["cover_type"].value_counts(); m.describe().T
```

Python 解释器：`C:\Projects\AIIAA\.venv\Scripts\python.exe`（已装 pandas、pyarrow、lightgbm、scikit-learn、statsmodels、matplotlib）。想用 Excel 看就导出小样本：`m.head(1000).to_csv("sample.csv")`。

## 2. 一行代表什么

保单表的一行是"一张保单在一个自然年内的一段保障期"。`exposure` 是这段期间的长度，单位是年。同一张保单年中有变化（换车、搬家）就会拆成多行；中途退保或不续保，这一行就提前结束。`policy_id` 跨年不变。

## 3. 各文件一览

| 文件 | 行数 | 列数 | 一行是 | 用途 |
|---|---|---|---|---|
| policies_motor | 979,968 | 39 | 保单期间 | 车险训练，2021–2024 |
| policies_buildings | 707,726 | 27 | 保单期间 | 房屋结构险训练 |
| policies_contents | 721,617 | 28 | 保单期间 | 家财险训练 |
| claims | 163,150 | 12 | 一笔理赔 | 训练期全部已报案理赔，评估日 2024-12-31 |
| test_motor | 254,986 | 38 | 保单期间 | 2025 车险，要预测 |
| test_buildings | 184,155 | 26 | 保单期间 | 2025 房屋结构险，要预测 |
| test_contents | 187,498 | 27 | 保单期间 | 2025 家财险，要预测 |
| customers | 545,240 | 18 | 客户 | 客户属性，跨险种共用 |
| addresses | 575,983 | 16 | 地址 | 地理位置与房屋特征 |
| weather | 2,879,915 | 9 | 地址×年 | 每个地址每年天气，含 2025 年 |
| sample_submission | 626,639 | 2 | 测试行 | 提交格式，prediction 全是 0 |

测试集比训练集少一列 `renewal_decision`。sample_submission 的 row_id 与三张测试表的 row_id 完全一致（已核对）。row_id 首字母：M 车险，B 房屋结构险，C 家财险。

## 4. 表之间怎么连

```
customers ──customer_id──┐
                         ├── policies_* / test_* ──policy_id + loss_date──> claims
addresses ──address_id───┤
weather ──address_id + 年份┘
```

- 理赔对保单行：同一 `policy_id`，且 `period_start ≤ loss_date ≤ period_end`。实测每笔理赔最多只对上 1 行，没有一对多。
- 天气按 `address_id` + 保单起始年份连接。测试集用 2025 年的天气。
- 一个客户可以同时有车险、房屋险、家财险；多个客户可能共用一个地址。
- 测试集中约 85% 的 policy_id 在训练集出现过（三个险种都是 84.8–84.9%），可以利用历史理赔经验。

## 5. 字段分组（详细定义见 references/kaggle_data_page.md）

**三个险种共有**：row_id, policy_id, customer_id, address_id, period_start, period_end, exposure, voluntary_excess（客户选的自愿免赔额 0/250/500/1000）, compulsory_excess（强制免赔额，车险固定 250，房屋/家财固定 350）, channel, payment_method, missed_payments_12m, new_business, renewal_count, days_quote_to_inception, discount_applied（0/0.05/0.10）, written_premium（现行定价收的保费）, customer_service_contacts, renewal_decision（仅训练集）。

**车险专有**：driver_age, years_licensed, ncb_years, vehicle_value_group(1–20), vehicle_age, make_model(300 种), engine_size_cc, body_type, transmission, seats, vehicle_colour, fuel_type, driver_assist, cover_type, annual_mileage, overnight_parking, vehicle_use, named_young_driver, telematics_opt_in, telematics_score。

**房屋结构险专有**：policyholder_age, sum_insured, construction_type, occupancy, days_unoccupied, smoke_alarms, accidental_damage_cover, mortgage。

**家财险专有**：policyholder_age, sum_insured, tenure, security_level, high_value_items, accidental_damage_cover, works_from_home, lodgers, bicycle_cover。

**customers**：出生日期、职业(20 类)、婚姻、家庭人数、信用等级 A–E、入保前 5 年理赔数、驾照扣分、在址年数、入保年份、邮箱域名、营销同意、文件偏好、前保险公司(40 家)、营销活动代码(51% 为空，表示无活动)、网点(25 个)、营销分群(12 类)。

**addresses**：区域 R01–R12、城市化程度、洪水区 1–5、离河距离、相对河面高程、土壤、建造年份、房屋类型、屋顶、地下室、5 米内有树、保护建筑、卧室数、能效等级、太阳能板。

**weather**：冰雹天数、风暴暴露指数、冰冻天数、夏季干旱指数、山火指数、年降雨量、年均温。

**claims**：line、peril（原因）、loss_date、report_date、status（closed/open/repudiated）、paid_to_date（已付，已扣免赔额）、incurred_to_date（已付 + 未决估损）、recoveries（追偿/残值回收）、repairer_id、third_party_id。

各险种的理赔原因：
- 车险：own_damage 48,964、third_party_prop 33,826、glass_hail 15,732、bodily_injury 7,583、theft 6,501
- 房屋结构险：escape_of_water 10,137、storm 7,816、accidental_damage 3,848、flood 2,232、subsidence 1,294、fire 1,115
- 家财险：accidental_damage 9,208、theft 7,863、escape_of_water 5,407、flood 1,624

## 6. 预测目标怎么从数据里算出来

官方定义：这一行期间内出险（loss_date 落在期间内）的全部理赔的**最终成本**，扣免赔额、扣追偿，不含拒赔。这是风险保费，不是收取的保费。

对应到数据：

```
每笔理赔净额 = incurred_to_date − recoveries      （status ≠ repudiated）
每行目标     = 对上这一行的所有理赔净额之和，再做发展调整到最终值
```

已核实的事实：
- 已结案理赔 incurred == paid，没有例外。
- 拒赔（repudiated）共 1,711 笔，paid 和 incurred 都是 0，删不删不影响金额，但要在文档里说明处理方式。
- incurred − recoveries 没有出现负数。
- 免赔额已经在 paid 里扣过了（官方写明 paid 是 net of excess）。

为什么要做发展调整（精算重点）：
- 评估日是 2024-12-31。2024 年出险的理赔中，未结案比例是车险 20.7%、房屋结构险 24.7%、家财险 13.0%，而 2021 年几乎为 0。
- 报案有延迟：车险报案延迟中位数 9 天，99 分位 377 天，最长 1361 天。2024 年末发生的事故有一部分还没报案（IBNR）。
- 车险净理赔总额：2021 年 1.06 亿、2022 年 1.17 亿、2023 年 1.17 亿、2024 年 0.91 亿。2024 年明显偏低，主要是没发展完，不是风险真的变低了。

## 7. 数据质量问题（已发现，建模前要处理）

| # | 问题 | 位置 | 规模 | 初步处理思路 |
|---|---|---|---|---|
| Q1 | exposure 有误：出现 365、289 等，疑似按"天"填；还有大量 1.0139 | 三个训练表 | 车险 3,991 行、房屋 3,044 行、家财 3,088 行 exposure > 1 | 用 (period_end − period_start + 1)/365 重算，并核对与原值的差异 |
| Q2 | 车险 customer_id 前导零丢失（如 `5180324`） | policies_motor、test_motor | 训练 98,110 行，测试 25,554 行 | 补齐到 8 位（zfill(8)）后 100% 能对上 customers |
| Q3 | sum_insured 有异常大值（如 2.2 亿），疑似放大约 1000 倍 | buildings、contents（训练和测试都有） | 房屋 1,395 行、家财 1,402 行 > 500 万 | 判定规则待定，需与同类保单的 written_premium 对照 |
| Q4 | construction_type 大小写混乱：Brick/brick/BRICK、stone/STONE、non_standard/NON_STANDARD，另有代码 W/w、T/t | buildings | 11 种写法，官方只有 brick/stone/timber/non-standard 4 种 | 统一小写。W 和 T 对应什么还没确认：测试集只有 W 没有 T |
| Q5 | 2020 年出险的理赔对不上任何保单行 | claims | 274 笔（车 198，房 46，家财 30），另有 2021–2023 年约 140 笔也对不上 | 不计入目标，在文档中记录 |
| Q6 | customer_service_contacts 训练集最大 36，测试集最大 6 | 训练表 | 车险 731 行 > 10 | 截断（capping）或检查是否与理赔有关（事后信息） |
| Q7 | 约 2.5% 的随机缺失：flood_zone、soil_type、roof_type、energy_rating、vehicle_age、engine_size、annual_mileage、overnight_parking、smoke_alarms、security_level、high_value_items、occupation、marital_status 等 | 多表 | 每列约 2.5% | 增加"缺失"类别，或交给树模型处理 |
| Q8 | telematics_score 约 87% 为空 | 车险 | 未开通设备的人都是空值 | 属于结构性缺失（开通了才有值），不要填补 |
| Q9 | renewal_decision 只在训练集，是这一行结束之后才知道的信息 | 训练表 | 全部 | 不能用作特征（数据泄漏） |
| Q10 | 训练集与测试集分布有差异：renewal_count 中位数 6 → 3，vehicle_age 中位数 7 → 9，ncb_years 中位数 7 → 8 | 训练 vs 测试 | 整体 | 在验证方案和报告中说明 |

## 8. 评分口径

Tweedie D²，power = 1.5，以常数预测为基准，越高越好。所有预测必须 ≥ 0。提交时三个测试表合并成一个文件，表头为 `row_id,prediction`。
