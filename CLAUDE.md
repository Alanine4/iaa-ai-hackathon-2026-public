# AIIAA：IAA AI Assisted Hackathon 2026

精算定价比赛：预测 2025 年车险、房屋结构险、家财险每个保单行的最终理赔成本（Tweedie D²，power 1.5）。

## 每次会话开始先读
1. `TODO.md`：进度清单。做完一项立即把 `[ ]` 改成 `[x]`，写上日期和产出文件。
2. `docs/COMPETITION.md`：比赛流程、评分、规则。
3. `docs/DATA_GUIDE.md`：数据结构、目标定义、数据质量问题 Q1–Q10。
4. `docs/references/`：Kaggle 网页原文存档，是权威来源。与其他文档冲突时以这里为准。

## 约定
- Python 只用 `.venv\Scripts\python.exe`；不用 R。
- 原始数据在 `data/`，只读，不改。
- 可复用的逻辑放在 `src/aiiaa/`（clean、target、development），先写测试，测试放在 `tests/`；运行测试：`.venv\Scripts\python.exe -m pytest -q`
- 一次性分析脚本放在 `scripts/`，分析输出放在 `docs/`，生成的中间数据放在 `output/`（不进 git）。
- 每一处修改都要记到 `docs/CHANGELOG.md`（编号 C0xx），并在代码注释里引用这个编号。代码和注释一律用英文。
- 读取标识符列（customer_id、address_id、policy_id、row_id、claim_id）时一律当文本处理。
- 每一步 AI 参与的工作都记到 `docs/AI_LOG.md`，写清人工如何核对。
