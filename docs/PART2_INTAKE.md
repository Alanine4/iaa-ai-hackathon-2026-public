# Part 2 数据入库说明（2026-10-24）

这份文档把 Part 2 下载下来的数据先过一遍：里面有什么、和 Part 1 一不一样、新市场哪里不同、有哪些脏数据、会动到现有流程的哪些部分、接下来建议做什么。所有数字来自 2026-10-24 实际跑出的结果，出处写在每节末尾。官方字段说明以 `docs/references/part2_kaggle_data_page.md` 为准。

## 1. 数据包含什么

位置：`data/iaa-ai-assisted-hackathon-2026-part-2/`，里面分两个文件夹。

- `developed/`：发达市场，11 个文件，官方说是 Part 1 的原数据，供参考和复用。
- `emerging/`：新兴市场，13 个文件。比 `developed/` 多两个文件：`expenses.parquet`（费用假设）和 `reference_policies.parquet`（20 张样例保单，用来写报告里的"营业保费"推导）。

`emerging/` 各文件的规模：

| 文件 | 行数 | 说明 |
|---|---|---|
| policies_motor | 47,283 | 车险训练，2023–2024 |
| policies_buildings | 27,447 | 房屋结构险训练 |
| policies_contents | 34,566 | 家财险训练 |
| claims | 9,258 | 车 7,183、房屋 1,059、家财 1,016，评估日 2024-12-31 |
| test_motor / test_buildings / test_contents | 90,521 / 52,295 / 66,247 | 2025 年，要预测的行，合计 209,063 |
| customers / addresses | 135,856 / 136,914 | 区域 E01–E06 |
| weather | 410,742 | 每个地址 2023、2024、2025 三年（官方表格写的是 2023–2024，但文件里有 2025 年，和 Part 1 的做法一样，测试集要用 2025 年） |
| expenses | 8 | 见第 6 节 |
| reference_policies | 20 | 取自测试文件，7 车险、7 房屋、6 家财 |
| sample_submission | 209,063 | 注意是 parquet 格式，Kaggle 要交的是 CSV（表头 `row_id,prediction`），只交新兴市场的行 |

和 Part 1 比，训练数据少得多：车险 4.7 万行，Part 1 是 98 万行。测试行数反而是训练行数的约两倍（车险 9.05 万 vs 4.73 万），因为 2025 年有一个大的经纪人业务包进来。

已做的备份：`emerging/MANIFEST.sha256`（13 个文件）和 `developed/MANIFEST.sha256`（11 个文件）已写好，文件已设为只读（C007 的做法）。目前 `tests/test_raw_data_integrity.py` 只覆盖 Part 1 文件夹，Part 2 还没有对应的完整性测试。

出处：`docs/part2_emerging_profile.txt`。

## 2. developed 是否与 Part 1 一致

一致，逐字节相同。`developed/` 的 11 个文件的 SHA256 与 `data/iaa-ai-assisted-hackathon-2026-part-1/MANIFEST.sha256` 逐个比对，11 个全部相同（addresses、claims、customers、policies_buildings、policies_contents、policies_motor、sample_submission、test_buildings、test_contents、test_motor、weather）。文件大小也一样。所以 Part 1 的代码、测试、结论可以原样用在 developed 上，不需要重新清洗或重新建模。

一个要当心的坑：两个市场里的编号会撞。`customer_id` 有 752 个相同、`address_id` 有 8,046 个相同、训练表的 `row_id`（M00000000 起）47,283 个完全相同，`policy_id` 也有少量相同，但它们是互不相关的记录。官方也说了不能跨市场连表。以后如果把两个市场的数据拼在一起训练，必须先加一列市场标记，编号不能直接当键用。测试集的 `row_id` 在两个市场之间没有重复。

## 3. emerging 与 developed 的主要差异

对比口径：2023–2024 两年合并，发达市场也只取 2023–2024 年，未做发展调整（两个市场评估日都是 2024-12-31，所以 2024 年同样没发展完）。完整表格见 `docs/part2_market_comparison.md`。差异超过 25% 的标"是"。

### 3.1 业务表现

| 项目 | 险种 | developed | emerging | 差异 | 超 25% |
|---|---|---|---|---|---|
| 理赔频率（次/风险年） | 车险 | 13.48% | 18.09% | +34% | 是 |
| | 房屋结构险 | 4.46% | 4.50% | +1% | 否 |
| | 家财险 | 4.05% | 3.43% | −15% | 否 |
| 平均每笔理赔金额（CU） | 车险 | 3,862 | 2,250 | −42% | 是 |
| | 房屋结构险 | 9,874 | 9,052 | −8% | 否 |
| | 家财险 | 2,886 | 2,589 | −10% | 否 |
| 已报告的纯保费（CU/风险年） | 车险 | 521 | 407 | −22% | 否（接近） |
| | 房屋结构险 | 441 | 407 | −8% | 否 |
| | 家财险 | 117 | 89 | −24% | 否（接近） |
| 保费中位数（CU） | 车险 | 493 | 502 | +2% | 否 |
| | 房屋结构险 | 228 | 238 | +4% | 否 |
| | 家财险 | 134 | 119 | −11% | 否 |
| 训练期风险年 | 车险 | 399,468 | 38,969 | 约 1/10 | 是 |
| | 房屋结构险 | 299,905 | 23,468 | 约 1/13 | 是 |
| | 家财险 | 299,942 | 29,058 | 约 1/10 | 是 |
| 训练期理赔笔数 | 车险 | 53,854 | 7,048 | | |
| | 房屋结构险 | 13,388 | 1,056 | | |
| | 家财险 | 12,146 | 998 | | |

要点：

- 车险是新市场差别最大的险种：出险更频繁，但每笔更小，两者大致抵消，纯保费低 22%，而现行保费却和发达市场差不多（中位数 502 vs 493）。
- emerging 自己的 2024 年纯保费比 2023 年低：车险 433 → 383，房屋 514 → 307，家财 91 → 87。2024 年没发展完，房屋只有 500 多笔理赔，波动很大，不能直接当趋势。
- 理赔的报案速度和未结案比例与发达市场接近：2024 年出险的未结案比例车险 20.0%（发达 20.7%）、房屋 25.3%（24.7%）、家财 11.5%（13.0%）；报案延迟中位数 9/8/6 天（发达 9/9/6 天），99 分位车险 331 天（377）、房屋 182 天（239）、家财 72 天（63）。

### 3.2 客户与保单结构

| 项目 | developed | emerging | 备注 |
|---|---|---|---|
| 渠道（车险为例） | 经纪人 25%、比价网站 35%、直销 40% | 经纪人 70%、比价网站 5%、直销 25% | 三个险种都一样，差异超 25% |
| 测试集中新业务占比 | 15% | 76–77% | 训练集里新业务只占 8%（车险约 3,040 风险年） |
| 测试保单在训练集出现过的比例 | 85% | 24% | |
| 年龄段（车险，17–24 岁占比） | 13% | 训练 20%，测试 25% | 测试集比训练集更年轻：25–34 岁训练 19%、测试 25%，65 岁以上训练 29%、测试 24% |
| 租户占比（家财险） | 30% | 43% | |
| 电动车占比 | 8.1% | 2.0% | |
| 车龄中位数 | 7 年 | 9 年（测试 10 年） | |
| 驾龄中位数 | 28 年 | 24 年（测试 13 年） | |
| 缺失率 | 约 2.5% | 约 12% | 同一批列，例如 vehicle_age、overnight_parking、soil_type、occupation |
| 天气（2024 年均值） | 冰雹 6.8 天、冰冻 39.5 天、野火指数 0.16、降雨 1,036 mm | 冰雹 17.1 天、冰冻 17.5 天、野火指数 0.55、降雨 1,375 mm | 房屋险暴风雨占理赔比例 44%（发达 30%） |
| 区域 | R01–R12 | E01–E06 | 编码不通用 |
| 房屋类型、城市化程度、构造类型占比 | | | 两个市场基本一样，没有超 25% 的 |

经纪人业务包进来之后，测试集的渠道、区域占比和训练集几乎一样（经纪人都是 70%），所以从这些字段看不出"新来的业务"和老业务有什么区别，主要区别在于新业务占比、年龄段、驾龄。

### 3.3 字段差异（官方说明）

| 表 | developed 有、emerging 没有 | emerging 有、developed 没有 |
|---|---|---|
| customers | date_of_birth、credit_band | gender（男女各约 50%，无缺失） |
| policies_motor / test_motor | driver_age、annual_mileage、telematics_opt_in、telematics_score | driver_age_band（17-24、25-34、35-49、50-64、65+） |
| policies_buildings / contents 及测试 | policyholder_age | policyholder_age_band |
| construction_type | brick、stone、timber、non_standard（大小写混乱） | BRK、STN、WOD、MIX（全是大写，干净） |
| occupation | 20 种规范写法 | 自由文本，80 种原始写法 |

出处：`docs/part2_market_comparison.md`、`docs/part2_emerging_quality.txt`。

## 4. 发现的数据质量问题（新旧对照表）

Part 1 的编号 Q1–Q10 见 `docs/DATA_GUIDE.md` 第 7 节。完整输出在 `docs/part2_emerging_quality.txt`。

### 4.1 Part 1 的问题在 emerging 里还在不在

| 编号 | Part 1 的问题 | emerging 现状 | 数量 |
|---|---|---|---|
| Q1 | exposure 出错（365、1.0139 等） | 还在，只在训练表；测试表 exposure 全部正确（0 行 > 1）。错的有三种：1.0139（等于 365/360，按 360 天算的整年）、直接填天数（如 365）、按 360 天算的部分年（如 302 天填成 0.8389 = 302/360） | 车险 253 行（0.53%：360 天口径 206 行含 141 行 1.0139，直接填天数 47 行）；房屋 153 行；家财 198 行。用 (结束日 − 开始日 + 1)/365 重算即可，规则不变 |
| Q2 | 车险 customer_id 丢前导零 | 不存在。所有表的 customer_id 都是 8 位，全部能连上 customers（0 行连不上），13,490 个以 0 开头的客户编号完整保留 | 0 |
| Q3 | sum_insured 被放大约 1000 倍 | 还在，规则可用。正常值最大：房屋 813,000（训练）/894,000（测试），家财 235,000/283,200；异常值最小：房屋 60,000,000，家财 7,600,000，两者之间有明显空档，阈值 500 万依然分得开 | 房屋训练 53 行、测试 104 行；家财训练 57 行、测试 135 行（约 0.2%） |
| Q4 | construction_type 大小写混乱、W/T 代码 | 写法变了：只有 BRK、STN、WOD、MIX 四种，全大写，没有大小写问题；但现有 `fix_construction_type` 不认识这些代码，会直接报错 | 训练 27,447 行、测试 52,295 行全部要映射 |
| Q5 | 理赔对不上保单行 | 还在，但很少：全是出险日早于该保单第一行的理赔（34 笔出险于 2021–2022 年，2 笔出险于 2023 年；市场 2023 年才开始写），没有出险日晚于保单最后一行的 | 36 笔（车 28、房屋 3、家财 5），共 92,268 CU，占理赔总额 0.3%；没有一笔理赔对上多行 |
| Q6 | customer_service_contacts 训练集最大 36、测试集最大 6 | 同类现象还在：训练最大 20，测试最大 5；该列已在 C008 删除，不影响 | 训练行 > 10：车险 61、房屋 2、家财 1 |
| Q7 | 约 2.5% 随机缺失 | 比例变高：约 12%（campaign_code 20%，Part 1 是 51%）。flood_zone 仍是约 3% | 例如 occupation 12.16%、vehicle_age 11.85%、soil_type 12.04%、distance_to_river_km 11.99%（Part 1 是 2.5%） |
| Q8 | telematics_score 87% 为空 | 不适用：telematics_opt_in、telematics_score、annual_mileage 在 emerging 里根本没有这些列 | |
| Q9 | renewal_decision 只在训练集（泄漏） | 一样：训练表有，测试表没有，仍不能当特征 | 三个训练表全有 |
| Q10 | 训练集与测试集分布差异 | 大得多，见下面 | 见第 3.2 节 |

Q10 的具体情况：测试集新业务占 76–77%，训练集只占 8%；renewal_count 在训练集中位数 7、最大 16，测试集中位数 0；而且在这份数据里 renewal_count = 0 与 new_business = 是 完全等价（0 行例外）。

### 4.2 emerging 新发现的问题

| 编号 | 问题 | 数量 | 说明 |
|---|---|---|---|
| N1 | occupation 是自由文本 | 80 种原始写法，119,339 个非缺失值 | 前后空格（如 `Retired ` 带尾空格）和大小写混用：去空格、转小写后剩 40 种；这 40 种里一半是 20 种基本职业，另一半是加了前缀 `Self-employed ` 的同名职业（共 11,975 条，占 10%）。有些组合不合常理（`Self-employed retired`、`Self-employed student`、`Self-employed unemployed`）。Part 1 只有 20 种规范写法 |
| N2 | expenses 里 emerging 的比价网站佣金是空白，但数据里有比价网站渠道 | 比价网站渠道占训练行 4.9–5.2%、测试行 5.0% | 官方说空白表示"不适用"，与数据矛盾。需要自己定一个假设（见第 6 节） |
| N3 | renewal_count 和 customer_since_year 与"该公司只写了两年"矛盾 | renewal_count 最大 16；20.2% 的客户 customer_since_year 早于 2023（最早 2008） | 这两个字段不能理解成"在本公司的续保年数"，只能当作经纪人业务带来的历史年数。合成数据的产物，用于建模时要小心 |
| N4 | 部分理赔的出险日早于市场开业 | 见 Q5 的 36 笔 | 与 Q5 同一批 |
| N5 | 年龄只给年龄段，没有出生日期 | 全部 | 现有 `customer_age` 无法计算（见第 5 节） |
| N6 | ncb_years 大于 years_licensed | 训练车险 7,756 行（16.4%），测试 20,051 行 | 发达市场也有（约 13%），不是 emerging 独有，说明这类不一致来自数据生成方式，不需要在 emerging 单独处理 |

另外检查过、没有发现问题的项目：row_id 全部唯一；没有完全重复的行；同一保单的行没有时间重叠；没有跨年的行；理赔的报案日不早于出险日、不晚于评估日；已结案理赔 incurred = paid（0 例外）；incurred − recoveries 没有负数；拒赔（120 笔）paid 和 incurred 都为 0；age band、性别、区域没有大小写或空格变体（只有 occupation 有）；所有标识符格式正确；样例提交文件的 row_id 与三张测试表完全一致。

## 5. 对现有流程的影响

按组件逐个说，标注"要改"、"可能要改"、"不用改"。这里只是评估，我没有动任何代码。

| 组件 | 影响 | 判断 |
|---|---|---|
| `src/aiiaa/clean.py` 的 `fix_exposure` | 训练表仍有错 exposure，重算规则（天数/365）仍然正确 | 不用改 |
| `fix_customer_id` | emerging 的编号本来就是 8 位，补零不产生变化 | 不用改（无害） |
| `fix_sum_insured` | 500 万阈值仍然分得开（见 Q3） | 不用改 |
| `fix_construction_type` | 遇到 BRK、STN、WOD、MIX 会抛 ValueError | 要改：加一张 emerging 映射表 BRK→brick、STN→stone、WOD→timber、MIX→non_standard（官方说明："MIX = mixed or non-standard"）。建议按市场分开处理，不要把 W/T 的老规则和新代码混在一张表里 |
| occupation 清洗 | 现有代码没有任何文本整理。车险特征里还在用 occupation（房屋、家财已在 C026 删掉），80 种原始写法会被当成 80 个类别 | 要改：去空格、转小写；是否拆出"自雇"标记还是合并为 20 种，需要你决定 |
| `src/aiiaa/features.py` 年龄字段 | `customer_age` 由 date_of_birth 计算，emerging 没有这一列，会报错；driver_age、policyholder_age（数值）换成了年龄段（类别）；credit_band、annual_mileage、telematics_* 在 emerging 不存在 | 要改：需要一个"两个市场共用的年龄表示"。办法之一是把 developed 的数值年龄按同样的 5 个年龄段切开（`docs/part2_market_comparison.md` 就是这样对比的）；gender 只有 emerging 有 |
| features 里的区域等 | region 编码 R01–R12 与 E01–E06 不通用；`customer_since_year`、`renewal_count` 的含义见 N3 | 可能要改 |
| `src/aiiaa/development.py` 报案曲线 | 现在的曲线用发达市场 2021 年出险的理赔估计（那一年已观察满 3 年以上）。emerging 只有 2023、2024 两个出险年：2023 年观察满 12–24 个月，2024 年不足 12 个月；2021–2022 年只有 36 笔 | 要决定：emerging 数据估不出长尾，要么沿用 developed 的曲线（emerging 的报案延迟和未结案比例与 developed 接近，见第 3.1 节），要么用 emerging 的 2023 年出险数据做一次对照检验后再决定 |
| 趋势估计（`evaluate.trend_factor`，C009） | 现在用 2021–2024 四年外推到 2025。emerging 只有两个点，且 2024 年没发展完（房屋 2023→2024 报告纯保费 −40% 主要是没发展完和理赔太少），两点直线外推到 2025 不可靠 | 要改：趋势不能由 emerging 两个点单独估，需要决定是借用 developed 的趋势、设为 0，还是别的做法 |
| 大额封顶（C014，99.5 分位） | 按 emerging 自己算：车险封顶 27,578（只有 36 笔超过）、房屋 220,808（6 笔）、家财 28,312（6 笔）。发达市场分别是 51,585、147,445、28,449。房屋的封顶线只由 6 笔理赔决定，不稳 | 要决定：用 emerging 的 99.5 分位，还是借用 developed 的赔款分布形状。大额回补系数（总额/封顶后总额）也只靠这点样本估计 |
| 验证方案和权重搜索（2 折时间验证，两折都不劣于等权才采用，0.002 门槛） | 现在的折是 "2021–22 训练 → 2023 验证" 和 "2021–23 训练 → 2024 验证"。emerging 只能切出一折（2023 训练 → 2024 验证），而且 2024 年没发展完。"两折都不劣于等权"的规则没法执行 | 要改：需要重新定义验证方案和采纳规则，并在看结果之前定好（避免过拟合，见 memory 里的约定） |
| 调好的超参数（`params.py`、C017/C032） | 都是按百万行规模调的；emerging 车险只有 4.7 万行，房屋 2.7 万行，每个叶子的最小样本数等设置可能偏大 | 可能要改 |
| 目标变量（`target.py`） | 逻辑不变，对 emerging 同样适用（扣免赔额、扣追偿、去拒赔） | 不用改 |
| 测试行的预测 | 测试行 exposure 正确，不需要重算；76% 是新业务，训练里新业务只有约 3,040 风险年（车险） | 不用改代码，但模型对"新业务"的估计很不稳，是主要风险 |
| `expenses.parquet` | 不影响 Kaggle 分数，用于报告里的营业保费推导 | 见第 6 节 |
| `reference_policies.parquet` | 20 张保单的 row_id 全部在测试文件里，保费与测试文件里的值一致（20/20） | 见第 6 节 |

## 6. expenses 与 reference_policies 的用途

两个文件都不影响 Kaggle 分数，官方说明它们用于报告：对 20 张样例保单，说明怎样从你预测的风险保费，用费用假设推到营业保费（office premium）。由评委评审，写进报告。

`expenses.parquet` 全文：

| 项目 | developed | emerging | 单位 |
|---|---|---|---|
| fixed_expense_per_policy（每张保单固定费用） | 45.00 | 85.00 | CU |
| variable_expense（变动费用） | 0.12 | 0.15 | 占保费比例 |
| commission_direct（直销佣金） | 0.00 | 0.00 | 占保费比例 |
| commission_comparison_site（比价网站佣金） | 0.10 | 空白 | 占保费比例 |
| commission_broker（经纪人佣金） | 0.15 | 0.20 | 占保费比例 |
| premium_tax（保费税） | 0.12 | 0.08 | 占保费比例 |
| cat_reinsurance（巨灾再保） | 0.06 | 0.12 | 占财产险风险保费比例 |
| target_profit_margin（目标利润率） | 0.05 | 0.08 | 占保费比例 |

读这张表时要先决定几件事（我没有替你决定）：

- 比价网站佣金在 emerging 是空白，但 emerging 数据里有 5% 左右的比价网站渠道（N2）。20 张样例保单里没有比价网站渠道的（17 张经纪人，3 张直销），所以样例本身不受影响，但报告里需要说明这条的处理。
- 巨灾再保的单位是"占财产险风险保费"，所以应该只加在房屋结构险和家财险上，车险不加。
- 固定费用是按保单，不是按行。样例保单里有 exposure 只有 0.07、0.11 的行，是否按 exposure 折算需要定一个口径。
- 各项比例是相加后除以保费，还是各自乘以风险保费，以及保费税是否也对其他项目征收，官方没有写明，需要在报告里写清楚你的口径。

`reference_policies.parquet`：20 行，列比测试文件多一列 `line`（险种）和各险种列的并集（不属于该险种的列为空）。其中 18 张是新业务，17 张经纪人渠道，3 张直销。

## 7. 建议的下一步清单

我只列出，不替你排优先级；括号里是需要你拍板的点。

- [ ] 决定是否把 `emerging/MANIFEST.sha256`、`developed/MANIFEST.sha256` 和 Part 2 数据一起提交，并给 `tests/test_raw_data_integrity.py` 增加 Part 2 的校验（目前这些文件还没有提交）。
- [ ] 把处理流程改成"按市场参数化"（数据路径、清洗规则、字段表），为 `fix_construction_type` 增加 emerging 映射、为 occupation 增加整理规则；先写测试再改（TDD）。
- [ ] 决定建模思路（需要你拍板）：只用 emerging 数据、把 developed 模型直接套用并校准整体水平、还是两个市场合并并加市场标记。注意训练数据只有 4.7 万 / 2.7 万 / 3.5 万行。
- [ ] 统一年龄表示：把 developed 的数值年龄切成 5 个年龄段，去掉 emerging 没有的字段（date_of_birth、credit_band、annual_mileage、telematics_*）。
- [ ] 用 emerging 的 2023 年出险数据检验 developed 的报案曲线是否适用，再决定沿用还是重估。
- [ ] 决定趋势处理（需要你拍板）：借用 developed、设为 0，还是别的方式。
- [ ] 决定大额封顶（需要你拍板）：房屋只有 6 笔超封顶线。
- [ ] 重新设计验证方案和"采纳门槛"，并在看结果之前定好（需要你拍板）。
- [ ] 评估新业务的影响：测试集 76% 是新业务，而训练集的新业务只有约 3,040 风险年（车险），先看新业务与老业务的频率差异有多稳。
- [ ] 为 20 张样例保单写营业保费推导：先定比价网站佣金、巨灾再保适用范围、固定费用折算方式的口径（第 6 节）。
- [ ] 提交时注意：Week 2 排行榜只交新兴市场测试行的预测，格式是 CSV `row_id,prediction`，预测值 ≥ 0；sample_submission 在文件夹里是 parquet，要转成 CSV。Part 2 截止时间以 Kaggle 页面倒计时为准（抓取页面时显示还有 7 天）。

---

## 本文用到的文件

- `docs/part2_emerging_profile.txt`：emerging 每个文件的行数、列、类型、缺失率、类别计数、数值分布；expenses 与 reference_policies 全文
- `docs/part2_emerging_quality.txt`：数据质量检查完整输出
- `docs/part2_market_comparison.md`：两个市场的对比表（英文）
- 生成这些文件的脚本：`scripts/part2_profile.py`、`scripts/part2_quality.py`、`scripts/part2_compare.py`（都以文件夹路径为参数）
- 完整性清单：`data/iaa-ai-assisted-hackathon-2026-part-2/emerging/MANIFEST.sha256`、`data/iaa-ai-assisted-hackathon-2026-part-2/developed/MANIFEST.sha256`
