# DeepSeek Flash 工地安全快速检测

当前版本使用 DeepSeek Flash 对施工现场图片进行单次视觉安全检测。模型负责识图、违规判断并提出法规候选；Python 随后用本地已核验现行法规目录校验候选，输出模型候选、最终核验条款、匹配方法以及每条风险在原图上的范围框坐标。法规目录不参与图片识别，也不取代大模型的安全判断。

`scripts/make_group_meeting_ppt.py` 仅是旧五模型方案的历史 PPT 生成材料，不被检测脚本或 HTTP API 导入执行，也不代表当前系统架构。

坐标由模型按整图 0–1000 归一化网格估计，程序换算为原图像素的左上角 `x,y` 与 `width,height`。视觉模型定位是近似结果，无法可靠定位时返回 `null`，不得将估计框当作测量级边界。

当前支持的场景提示配置：

- 施工现场综合安全
- 临时用电/配电箱
- 高处作业
- 基坑工程
- 起重吊装
- 塔式起重机
- 施工升降机
- 扣件式钢管脚手架
- 承插型盘扣式钢管脚手架
- 悬挑式脚手架
- 附着式升降脚手架

## 0. 仓库结构

按"职责"分目录，每个目录只放一类东西：

```
safe-checker-v2-flash/
├── src/                     运行时代码（手写，不自动生成）
│   ├── safety_check.py        检测主程序，也是命令行入口
│   ├── api_server.py          HTTP 接口，视频处理端调用它
│   ├── scene_rules.py         场景档案、场景归一化、别名
│   ├── verified_regulations.py 已核验法规目录的装配与匹配
│   └── catalogs/              自动生成的目录（由 tools/ 重建，勿手改）
│       ├── rulepack_catalog.py    规则包的 63 条规则
│       ├── scene_checklists.py    场景 / 家族 / 主体三套检查清单
│       ├── standard_catalog.py    已抓取标准的 977 条条款
│       └── generated_rules.py     由标准条款反向生成的规则
│
├── data/                    规则与法规数据（只读输入）
│   ├── rulepack/              规则包快照
│   ├── main_system/           主系统导出的 1036 条规则与 335 个主体
│   ├── standards/            21 部标准的全文与清洗结果
│   └── generated/            反向生成规则的原始产出、中间产出与报告
│
├── tools/                   生成与维护脚本（改数据后跑这里的脚本）
├── tests/                   离线测试（97 项，不调用模型）
├── scripts/                 辅助脚本：远程部署、图像输入对比、组会 PPT
├── docs/                    流程说明、实验记录、提示词分段指纹
└── outputs/                 每次检测生成的报告（不入库）
```

三条约定：

1. **`src/catalogs/` 里的文件都是生成的**，改规则要改 `data/` 或 `tools/`，然后重跑对应脚本；
2. **`data/` 是输入，`tools/` 改数据，`src/` 只读它们**，三者不互相依赖；
3. **生成必须可复现**——同样的输入重跑一次，产出的文件逐字节相同。

## 1. 环境要求

- Python 3.10 或更高版本
- VS Code
- DeepSeek API Key

## 2. 安装环境

在 VS Code 终端中进入本目录：

```bash
cd "/Users/gjz/Documents/ChatGPT/复杂环境安全巡检/deepseek_flash_safety_checker"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. 配置 API

复制环境变量模板：

```bash
cp .env.example .env
```

打开 `.env`，填写自己的 API Key：

```env
DEEPSEEK_API_KEY=sk-你的真实密钥
DEEPSEEK_MODEL=deepseek-flash
REQUEST_TIMEOUT_SECONDS=8
DEEPSEEK_VISION_TIMEOUT_SECONDS=7.4
MAX_IMAGE_EDGE=768
IMAGE_DETAIL=low
SINGLE_MODEL_MAX_TOKENS=2200
REASONING_EFFORT=none
```

如果你的 DeepSeek Flash API 地址或模型名和默认值不同，请按实际平台文档修改：

```env
DEEPSEEK_BASE_URL=https://api.deepseek.com/chat/completions
DEEPSEEK_MODEL=deepseek-flash
```

## 4. 运行检测

先用 dry-run 验证图片和配置，不会调用 API：

```bash
python src/safety_check.py "/图片/绝对路径.jpg" --scene "起重吊装" --dry-run
```

普通运行（默认输出并保存标准 JSON）：

```bash
python src/safety_check.py "/图片/绝对路径.jpg"
```

如果已知场景，建议加上 `--scene`，这样更快、更准：

```bash
python src/safety_check.py "/图片/绝对路径.jpg" --scene "起重吊装"
```

常见场景示例：

```bash
python src/safety_check.py "/图片/绝对路径.jpg" --scene "施工现场临时用电"
python src/safety_check.py "/图片/绝对路径.jpg" --scene "高处作业"
python src/safety_check.py "/图片/绝对路径.jpg" --scene "基坑工程"
python src/safety_check.py "/图片/绝对路径.jpg" --scene "吊装作业"
python src/safety_check.py "/图片/绝对路径.jpg" --scene "塔吊"
python src/safety_check.py "/图片/绝对路径.jpg" --scene "施工升降机"
python src/safety_check.py "/图片/绝对路径.jpg" --scene "脚手架"
```

### 4.0 紧凑输出模式（八秒内完成率的关键）

`OUTPUT_MODE=compact`（默认）让模型用短键输出结构化结果，程序再把短键还原成完整字段。
语义规则与提示词正文逐字未改，只替换输出格式模板与法规字段要求。

| 模式 | 模型输出体积 | 94 张八秒内有效报告率 |
|---|---|---:|
| `full` | 约 4300 字符 | 48.9% |
| `compact` | 约 2000 字符 | **98.9%** |

配套的两项改动：

- 模型只写 `regulation.standard_code` 与 `regulation.article`，标准全称和条文原文由本地目录回填；
- 新增分级 JSON 容错（去尾逗号、转义内嵌引号、括号配平、逐条抢救），模型输出偶发不合法时
  保住可解析的风险项而不是整份丢弃。

最终报告的标准 JSON 结构与改动前一致，只新增输入策略相关元数据字段。

### 4.0.0 当前实测水平（2026-10-28）

项目测试集 94 张，深海道 flash，单模型单次调用，8 秒预算，同一时间窗内交错配对运行。

| 用法 | 命中率 | 报告完成率 | 中位耗时 |
|---|---:|---:|---:|
| 指定场景 `--scene`（推荐） | **85.9%**（79/92） | 97.9% | 5.32s |
| 不指定场景，AUTO 主体清单 | 66.3%（57/86） | 91.5% | 5.34s |

两个时间窗独立跑出来的指定场景命中数都是 79（84.0% / 85.9%），说明这条路径是稳定的。

知道场景就传 `--scene`，这是最快也最准的用法。

**关于"过度检查"的实测结论**：94 张共报出 201 条风险，其中 122 条不是测试集标注的那个缺陷，
对它们单独判定后 **75.4% 实际成立**；被当成"明确违规"报出来但实际不成立的只有 3 条（占全部 1.5%）。
所以当前版本的主要矛盾不是误报，而是**漏检**：15 张没报出标注缺陷，集中在
材料破损类（钢管弯折、冲压件折裂）与节点连接脱开类。详细分析见
`output/P10_过度检查与漏检的真实结构_20261028.md`。

### 4.0.1 瞬时故障重试

检测器与 API 层都区分"瞬时故障"和"确定性失败"，前者会在预算允许时重试：

| 层级 | 机制 | 默认 |
|---|---|---|
| 检测器内部 | `call_deepseek` 共享同一预算多次尝试；DNS/连接类失败只花约 0.24 秒，可用剩余预算再试；读超时已耗尽预算，不重试 | `MODEL_MAX_ATTEMPTS=3`、`MODEL_RETRY_MIN_BUDGET=1.0`、`MODEL_CONNECT_TIMEOUT=2.0` |
| API 外层 | 对 `timeout` 与 `network_error` 整体重跑一次，受 API 自身总预算约束 | `API_TRANSIENT_RETRIES=1`、`API_TOTAL_TIMEOUT_SECONDS=17.5` |

连接阶段超时单独收紧，让 DNS 失败尽快暴露；进程硬杀上限比检测预算宽 1.5 秒，
保证检测器能返回结构化超时错误而不是被 SIGKILL。
报告的 `timings.model_attempts` 记录实际尝试次数。

**账户余额或限流（HTTP 402 / 429）单独归类为 `insufficient_balance`**，不重试、也不算一次检测失败：
这类响应说明请求没进入推理，调用方需要充值后重试，不能把它记成"该图没检出"。

重试提升的是"交付率"：单次调用的完成率仍随服务端状态波动（实测 65%–99%），
允许重试后稳定在 95% 以上。

### 4.0.2 场景检查清单

`tools/build_scene_checklist.py` 从规则包生成按场景组织的检查清单，写入 `src/catalogs/scene_checklists.py`。
规则包里同一条规则在不同场景重复出现（如劳保类规则），生成时按 `judge_dimension` 去重，
每个检查维度渲染成一行，附带违规／合规／存疑三态边界。

- 注入位置：语义段内，范围框段之后、法规要求段之前；
- 指定场景时给该场景完整清单；
- 已指定但尚未开放清单的场景不注入任何清单；
- 由现有「对象与状态协议」管辖的维度（积水、栏杆与网片封闭）渲染为指回该协议，
  避免清单的边界比协议更松；
- 清单段固定声明"只输出证据充分、确实成立的风险，不要为凑满清单而报告"。

当前全部 11 个场景已铺开（`ENABLED_SCENES` 列出全部场景键）。每个场景的清单按"场景专属性"排序：
只在该场景出现的检查维度排在最前，通用劳保维度排在最后，避免模型的注意力被通用项占满。

#### 主体锚点（默认关闭）

清单每一行前面可以加 `[主体名]`，主体名取自规则包的 `triggers` 字段，不是手写的：

```
- [架体作业层承脚与承放表面] 使用中的架体清理建筑垃圾及其它杂物｜违规:…
```

维度名只说"查什么、什么算违规"，不说"去哪儿看"，所以加锚点本来是想补上后半句。

**但 94 张配对实测结果是负的**：开 76/93（81.7%），关 79/94（84.0%），方向 −2.2%。
清单本身已经按场景组织，场景已知时主体名是冗余的，还会把每行撑长约 140 字符。
所以**默认关闭**，需要复现实验时用 `CHECKLIST_SUBJECT_ANCHOR=1` 打开。

#### 家族合并清单

模型把场景**族**判对的概率约 90%，判到**子类**只有 28–49%。
所以脚手架四类（悬挑式／扣件式／盘扣式／附着升降式）合成一个 `SCAFFOLD_FAMILY` 清单，
起重设备三类（塔式起重机／施工升降机／起重吊装）合成 `LIFTING_FAMILY`。

- 只在部分子类存在的维度会标注"仅XX适用"，子类定不下来时可以跳过，避免误报；
- 直接 `--scene "脚手架"` 会走家族清单（`SCAFFOLD_FAMILY`），
  `--scene "扣件式钢管脚手架"` 仍然走精确清单；
- `auto` 不再落到综合场景，`normalize_scene("auto")` 返回 `AUTO`。

#### AUTO（未指定场景）的三种给法

用 `CHECKLIST_AUTO_MODE` 切换，默认 `subject`：

| 取值 | 给法 | 清单字符 | 94 张命中率 |
|---|---|---:|---:|
| `subject` | 主系统 trigger→rule 机制的单次调用形态，62 个主体一行一条 | 4697 | **72.1%** |
| `scene` | 旧做法，逐场景给 3 条维度名称，不带边界 | 1256 | 67.7% |
| `family` | 按家族给核心维度，带三态边界 | 2169 | 62.9% |

三种给法在同一时间窗内交错跑了 470 次调用。**覆盖广度决定胜负**：
`subject` 覆盖 62 个主体（74 条规则-主体对），`scene` 覆盖 33 条维度，`family` 只覆盖 30 条。
家族清单虽然带了三态边界，却因为每族只留 5 个核心维度而损失覆盖，反而最差。

`subject` 模式的一行长这样：

```
- 主体[塔机吊钩与钩口] 吊钩防钢丝绳脱钩装置存在｜违规:完整吊钩防脱配置区域明确无该装置。
```

提示词要求模型"先逐条确认主体是否存在，只核对确实存在的主体的条目，主体不存在直接跳过"。
这条路绕开了场景分类这一步——局部主体判断比场景分类容易得多。
代价是提示词更长，且为了控制长度只保留了违规边界。

| 场景 | 规则条数 | 去重后维度 | 清单字符 |
|---|---:|---:|---:|
| 扣件式钢管脚手架 | 17 | 13 | 1056 |
| 悬挑式脚手架 | 15 | 11 | 860 |
| 临时用电/配电箱 | 15 | 11 | 950 |
| 承插型盘扣式钢管脚手架 | 14 | 10 | 769 |
| 起重吊装 | 12 | 8 | 780 |
| 塔式起重机 | 13 | 9 | 791 |
| 附着式升降脚手架 | 12 | 8 | 660 |
| 施工升降机 | 10 | 6 | 558 |
| 基坑工程 | 9 | 5 | 401 |
| 高处作业 | 9 | 5 | 425 |
| 施工现场综合 | 7 | 3 | 284 |
| 脚手架（家族合并） | 37 | 22 | 2337 |
| 起重与升降设备（家族合并） | 21 | 17 | 1901 |

家族合并清单的实测结论：**命中率没有提升**（子类精确清单 29/36、家族 29/39，命中数相同），
但它保留下来是因为子类不确定时更保险——盘扣式的检查项不会因为被归成扣件式而整条丢掉。
所以 `--scene "脚手架"` 走家族清单，`--scene "扣件式钢管脚手架"` 仍走精确清单。

规则包维度名里的"推荐条款："等痕迹与句末标点会在生成时清理。

`tools/prompt_baseline.py` 记录提示词各段落的指纹与归属方，改提示词后重跑并提交差异，
让并行的工作线只在自己负责的段落内追加，避免互相覆盖。

### 4.0.3 本地法规目录

本地目录由 `verified_regulations.py` 与 `rulepack_catalog.py` 组成：

- `verified_regulations.py`：GB 55034-2022（28 条）与 JGJ/T 46-2024（13 条），来自官方文件或二次来源；
- `rulepack_catalog.py`：本组规则包 `rulepack.sqlite3` 快照的 63 条规则，覆盖 14 部标准
  （GB51210-2016、GB5144-2006、GB50194-2014、JGJ130-2011、JGJ202-2010、JGJ 276—2012、
  JGJ311-2013、JGJ80-2016、JGJ184-2009、JGJ/T 231-2021、DG/TJ08-2002-2006、GB 10055—2007、
  GB/T34023-2017、GB/T5031-2019）。

规则包条目由 `tools/build_rulepack_catalog.py` 从 `data/rulepack/rulepack_snapshot.json` 生成，
来源标记为 `project_rulepack_snapshot`，与官方文件严格区分，引用前仍需核对现行版本。

第三部分 `standard_catalog.py` 是从公开可获取的标准全文副本中抽取的条款，由
`tools/download_standards.py` 抓取、`tools/build_standard_catalog.py` 生成，共 977 条、5 部标准。
每条都保留来源 URL、下载文件 SHA256、抽取方式与抓取时间；**只有与规则包共同覆盖条款的
一致率 ≥85% 的标准才会进目录**，未通过校验的条款一律不采用。来源标记为
`standard_fulltext_public_copy`。

当前目录合计 1081 条、21 部标准：

| 来源 | 条数 | 标记 |
|---|---:|---|
| 官方文件 | 28 | `official_primary_document` |
| 二次来源 | 13 | `secondary_reference` |
| 本组规则包快照 | 63 | `project_rulepack_snapshot` |
| 标准公开全文 | 977 | `standard_fulltext_public_copy` |

`tools/download_standards.py` 可重复运行；标准全文抓取结果与交叉校验结论见
`data/standards/cross_check.json`。

### 4.1 可选的全图与局部图实验

借鉴 Monkey 保留全局与局部细节的思路，可以在**同一次 DeepSeek Flash 请求**中发送原图和一张局部图。仍由模型完成安全判断，法规核验及标准 JSON 格式沿用主流程。这是输入策略实验，不是复现 Monkey 的模型训练。

默认 `DETAIL_CROP=off`，原来的运行指令仍使用单张全图。以下参数可覆盖 `.env`：

```bash
# 单张全图基线
python src/safety_check.py "/图片/绝对路径.jpg" --detail-crop off

# 自动选择一块纹理较丰富的区域，附加一张局部图
python src/safety_check.py "/图片/绝对路径.jpg" --detail-crop auto

# 指定局部框：原图归一化到 0–1000 的 x1,y1,x2,y2，非像素及非宽高
python src/safety_check.py "/图片/绝对路径.jpg" --detail-crop "140,180,790,930"
```

手工坐标必须在 `0..1000` 内，且右下角大于左上角；坐标系采用 EXIF 方向校正后的整图。裁剪从原始清晰图像中生成，全图与局部图分别缩放；模型输出的风险框仍以**整张原图**为参照，`issues[].bbox_original_px` 继续返回原图 `x,y,width,height`。程序不会猜测模型是否误用了局部图坐标，定位仍需验证。

`auto` 只使用边缘纹理的廉价启发式，不识别安全目标，可能选到复杂背景；当前“检测2”实验使用手工框诊断细节增益，不能把它当作自动选区精度。局部图不能恢复原照片的模糊和遮挡，也会增加视觉输入和时延。

可选配置：

```env
DETAIL_CROP=off
DETAIL_CROP_MAX_EDGE=768
```

结果新增 `input_strategy`、`image_inputs`、`vision_calls` 和 `detail_crop`，方便确认实际使用了单图还是双图。`vision_calls=1` 在成功结果中表示一次模型请求；dry-run 中只是计划值，不代表已请求模型。

### 4.2 可复现的多轮对照

先验证本地流程，不调用 API：

```bash
python scripts/compare_image_inputs.py "/Users/gjz/Pictures/检测2.png" \
  --scene "施工现场临时用电" --detail-crop "140,180,790,930" \
  --repeats 1 --dry-run
```

真实对照默认每组 3 次，共 6 次模型请求，逐次交错运行，无自动重试：

```bash
python scripts/compare_image_inputs.py "/Users/gjz/Pictures/检测2.png" \
  --scene "施工现场临时用电" --detail-crop "140,180,790,930" \
  --repeats 3 --budget 8
```

脚本在 `outputs/input_comparison/<时间>/` 保存逐次原始结果、`comparison.json` 和 `comparison.md`。分开记录预处理、模型、脚本内耗时与含 Python 启动的进程墙钟；超时和失败计入尝试数，单个子进程到预算会终止。stdout 为标准 JSON 摘要，进度写入 stderr。该计时不包含 HTTP 客户端上传及回传。

对照工具不自动给准确率打分，需人工核对风险、法规和范围框。单张图片、少量重复不能证明跨场景提升或稳定 p95 时延。自动选区可将同一指令的 `--detail-crop` 改为 `auto`。

## 5. 输出格式

终端会输出：

```text
报告格式版本：1.3
检测模型：DeepSeek Flash（单模型单次视觉检测）
检查场景：施工现场综合安全
原图尺寸：2000 × 1000 像素

1. 明确违规：……
• 目标位置：……
• 现象描述：……
• 违反的具体安全条例：《建筑与市政工程施工现场临时用电安全技术标准》JGJ/T 46-2024 第……条：……
• 违规范围框（原图像素，左上角原点）：x=200, y=200, width=600, height=450
• 单模型判断置信度：0.90

流程耗时：3.38 秒
```

普通运行默认把带缩进的标准 JSON 打印到终端，并保存到 `outputs/` 目录。JSON 不再重复嵌入整段 Markdown，风险、条例和范围框统一放在 `issues` 数组中，其他成员可直接使用 `issues[].bbox_original_px` 绘制范围框。请求使用接口原生 `response_format=json_object` 约束；超时、配置错误和处理异常也输出统一的 JSON 错误对象。只有需要纯文本报告时才使用 `--text-output`。

## 6. 八秒内响应的关键设置

`.env` 中这些参数会影响速度：

```env
REQUEST_TIMEOUT_SECONDS=8
DEEPSEEK_VISION_TIMEOUT_SECONDS=7.4
IMAGE_DETAIL=low
MAX_IMAGE_EDGE=768
JPEG_QUALITY=70
SINGLE_MODEL_MAX_TOKENS=2200
TEMPERATURE=0
REASONING_EFFORT=none
```

如果经常超过 8 秒，优先调小图片尺寸；人物清单版本不建议把输出上限降到 2000 以下，否则多人图片可能截断 JSON：

```env
MAX_IMAGE_EDGE=640
```

降低图片尺寸或输出 token 数可能减少识别细节或截断 JSON，应通过测试集验证后再长期使用。`high` 和 `max` 会启用更充分思考，通常更容易超过 8 秒。

深度思考强度可以这样设置：

```env
REASONING_EFFORT=none   # 关闭思考，最快
REASONING_EFFORT=low    # 最低思考强度，速度较慢
REASONING_EFFORT=high   # 更充分思考，速度较慢
REASONING_EFFORT=max    # 最大思考，速度最慢
```

当前任务如果要求 8 秒内稳定返回，建议使用：

```env
REASONING_EFFORT=none
```

原因是增强 Prompt 已经加入了证据约束和误判抑制，继续使用 `low/high` 可能把输出 token 消耗在思考内容上，导致超时或没有最终报告。

## 7. 误判抑制机制

### 7.1 模型候选与本地法规核验

DeepSeek Flash 在识别风险后，依据施工安全规范知识提出最可能的中国现行标准候选，并返回 `risk_category`、`risk_key`、规范全称、标准编号、条款号、条文要点和匹配状态。法规范围不局限于临时用电，可以覆盖高处、起重吊装、脚手架、基坑、机械、消防和个人防护等场景。

`verified_regulations.py` 只做法规真实性和版本核验，匹配顺序为：标准编号+条款号精确匹配、稳定风险键匹配、风险别名匹配、同场景多关键词候选匹配。每条可正式引用的记录必须具有完整规范名称、标准编号、具体条款原文、当前版本状态、来源链接和本地核验日期。关键词命中只能作为人工复核候选，不能直接支撑“明确违规”；模型候选仍保留在 `model_regulation_candidate` 中以便审计。匹配失败时风险项不会被删除，但明确结论会降级为疑似违规，`verified_regulation` 为 `null`，并要求人工核验。当前版本不是实时联网法规搜索，目录之外的条款不会被冒充为已核验结果。

每条风险还会返回 `regulation_status`：`VERIFIED` 表示已由本地目录核验；`MODEL_CANDIDATE_UNVERIFIED` 表示模型给出了完整候选，但未通过本地目录核验，不能作为正式引用；`UNMATCHED` 表示没有完整可靠的条例候选。`has_verified_regulation` 和 `regulation_verification_level` 可供视频端直接过滤；只有 `has_verified_regulation=true` 且视觉证据也满足门控时，报告才会使用“明确违规”和“违反的具体安全条例”。出现后两种状态不一定代表法规库遗漏，也可能是模型提出的要求并非该作业条件下普遍适用。例如一般施工人员穿着普通工作背心、白天照片中没有明显反光，不能单独证明违反了反光服装规定。

当前目录共 41 条已核验条款，覆盖 `JGJ/T 46-2024` 的临时用电高频可视风险，以及强制性通用规范 `GB 55034-2022` 中的人员防护、警示标识、外电线路、高处平台、临边洞口、坠物、吊装隔离、吊索具、斜拉斜吊、基坑排水、物料堆放、机械、施工升降机、施工车辆、受限空间、气瓶与易燃易爆品、水上作业和腐蚀品防护等内容。目录允许逐步扩充；“场景受支持”不等于“该场景全部条款已经入库”。

当前单模型检测链路：

1. DeepSeek Flash 完成图片识别和施工安全判断，代码不再调用其他视觉或文本模型。
2. Prompt 先建立 `objects` 对象/部件清单和 `persons` 人物清单，再判断对象身份、部件归属、装备状态与空间关系。
3. 开闭、缺失、积水、杂物和空间占用类结论必须返回 `claim_type` 与 `claim_validation`；代码对身份、边界、正向证据、反向解释和空间锚点进行一致性门控。
4. DeepSeek Flash 为每条风险给出稳定风险键和跨场景法规候选；代码使用本地目录进行精确或保守语义核验，关键词兜底只进入人工复核，不直接形成明确违规。
5. 违规范围框输出为原图像素坐标：左上角 `x,y`，以及 `width,height`；图像按 EXIF 方向校正后确定坐标系。
6. 同时输出模型候选、最终核验条款和匹配依据；无可靠匹配时保留风险并要求人工查询。

因此，图片受强光、反光、遮挡、模糊影响时，结果会更倾向于输出“疑似违规”或“不可判断”，减少把不确定内容写成确定违规。

对资质、票证、检测/维护记录、承载力、设计工况、基坑支护必要性等不能由单张图片独立证明的结论，代码只做通用证据降级，不替代模型生成新的违规项。手套、反光背心等任务型防护用品还必须同时具有可见作业动作、独立危险对象、对应危害和适用条款，不能仅因人员出现在工地、高处或车辆附近就判定缺失。模型风险描述本身不能反过来充当适用条件证据；例如模型写出的“可见度低”不能替代图片对象和人物动作中的真实可见证据。白天单张照片也不能根据背心颜色、反光条外观或没有出现强反光来判断材料反光性能不合格。

### 7.2 本地未命中与联网法规检索

主检测链路必须在约 8 秒内完成，因此当前请求不会临时访问搜索引擎。DeepSeek 返回的法规候选来自模型已有知识，不等同于实时搜索。若业务要求自动补齐目录外法规，推荐增加独立的异步法规核验服务：主链路先返回风险、视觉证据和 `regulation_status`；后台仅对 `MODEL_CANDIDATE_UNVERIFIED` 或 `UNMATCHED` 项检索国家法律法规数据库、住房和城乡建设部、应急管理部及国家标准全文公开系统等权威来源，校验标准状态、编号、条号、原文和适用条件后再回写结果。

联网检索结果必须经过结构化核验后才能升级为 `VERIFIED`，不能只凭搜索摘要或大模型自由生成内容作为正式条例。该异步设计既能逐步扩大法规覆盖面，也不会把网络搜索延迟计入图片检测的 8 秒预算。

8 秒控制方式：

```text
总超时 REQUEST_TIMEOUT_SECONDS=8
DeepSeek Flash 单次视觉请求 DEEPSEEK_VISION_TIMEOUT_SECONDS=7.4
剩余时间用于 JSON 解析、坐标换算和报告输出
```

模型请求使用 HTTPX 分阶段超时，并在其外增加整次请求的异步期限，覆盖连接、上传与完整响应等待。预处理后会重新计算剩余时间，扣除本地处理余量；预算用尽则返回标准 JSON 超时错误，不自动重试。单模型减少了多路并发等待，但进程启动、保存、HTTP 客户端上传回传及取消请求的收尾仍有额外开销，该时间要求不是任何网络情况下的成功保证。

## 8.1 高精度模型检查协议

当前 Prompt 要求视觉模型按以下顺序完成一次检查：

```text
对象清单
→ 主体设备识别、部件归属与场景证据
→ 为每名人员分配 P1/P2 编号
→ 头部、双手、躯干、腰部、双脚可见性
→ 装备佩戴位置、支撑面和离地状态
→ 人员与吊物、临边、设备的空间锚点关系
→ 多类别安全风险扫描
→ 规范匹配
→ 证据自检
```

每个模型风险项都要返回：

```json
{
  "target": "风险对象或人员编号",
  "visual_cues": ["相互独立的直接视觉线索1", "相互独立的直接视觉线索2"],
  "target_visibility": "complete/partial/occluded/not_visible",
  "evidence_level": "direct/partial/occluded/not_visible/ambiguous",
  "citation_confidence": "high/medium/low/unknown"
}
```

对象状态类风险还必须返回 `object_ids`、`claim_type` 和 `claim_validation`。例如主箱门打开必须同时具备主门门板角度位移和箱体开口/内部空间证据；积水必须具备连续水面及至少两项水体视觉特征；缺少护栏必须先完整检查上横杆、中横杆、立杆等构件；设备固定说明页不能判为杂物；空间占用必须证明两个边界实际重叠或阻断，而不是仅仅相邻。

例如，人员头部没有完整进入画面时，模型不能写“未戴安全帽”；左手套和右手套分别记录在 `hand_ppe_states` 中，只要对应手部被钢筋、工具或身体遮挡，“未戴手套”就会被程序降为“不可判断”。人物风险通过 `person_ids` 引用人物清单，代码会核对装备所对应的身体部位是否完整可见。空间风险还要返回身体锚点、参照物锚点、吊物状态和人员所在区域；依赖双脚位置的明确结论必须保证双脚完整可见，抬腿跨越低矮材料不会被保留为明确高处攀爬，靠近静置钢筋或钢丝绳也不会被保留为明确位于吊物下方。

钢筋装卸截图实测中，模型调用和全部本地处理总耗时分别为 `5.41 秒`、`7.33 秒`（网络响应存在波动），均低于 8 秒目标。两次均将左右手套标为 `uncertain`，未输出明确的“未戴手套”；收紧身体锚点门控后，脚部仅部分可见的“站在钢筋堆上”由明确违规降为疑似。该结果说明证据门控可以降低误报，但单张图片和两次调用不能证明所有工地场景都绝对准确，仍应使用标注测试集持续评估。

## 8. 对象状态与场景提示

结构化事实不再只有 `yes/no/uncertain`，还包括：

```text
not_applicable = 当前图片没有这个检查对象，或当前场景不适用
```

例如：

```json
{
  "distribution_box_visible": "no",
  "electrical_cable_visible": "no",
  "box_door_open": "not_applicable",
  "cable_ground_contact": "not_applicable"
}
```

当命令使用：

```bash
--scene "起重吊装作业"
```

场景配置只提供当前场景的检查字段和规范背景，不会把模型的安全检查限制为一个有限规则清单。模型仍需扫描图片中的其他清晰可见风险；同时 Prompt 明确要求不要把吊装钢丝绳、吊带或软管当成电气电缆。

## 9. 说明

- 脚本会自动压缩图片，减少上传和推理时间。
- 脚本根据 `REASONING_EFFORT` 设置思考模式：`none` 关闭，`low` 为最低思考强度。当前快速检测默认使用 `none`，优先保证 8 秒内稳定输出。
- 如果模型只返回思考内容而没有最终报告，程序会给出明确错误，不会保存空白报告。
- 如果图片不清晰或条款编号不确定，模型会输出“疑似违规”或“条款编号需复核”。
- 8 秒内完成依赖网络、图片大小和模型服务状态；脚本根据剩余预算设置整次模型请求期限，不能保证第三方服务必定在期限内完成有效检测。
- DeepSeek 图像输入文档：https://api-docs.deepseek.com/zh-cn/guides/vision/
- DeepSeek Thinking Mode 文档：https://api-docs.deepseek.com/guides/thinking_mode/

## 10. VS Code 运行

用 VS Code 打开本目录：

```bash
code "/Users/gjz/Documents/ChatGPT/复杂环境安全巡检/deepseek_flash_safety_checker"
```

然后进入 Run and Debug，选择：

- `Dry Run 检查配置`：不调用 API，只检查图片预处理和参数。
- `DeepSeek Flash 安全检测`：真实调用 API。

## 11. 视频处理端 HTTP API

项目提供 `src/api_server.py` HTTP API，视频处理端可上传图片并获取 DeepSeek Flash 单次检测结果及原图坐标框。视频端不需要、也不应传递模型厂商的 API Key。

安装依赖并启动服务：

```bash
cd "/Users/gjz/Documents/ChatGPT/复杂环境安全巡检/deepseek_flash_safety_checker"
source .venv/bin/activate
pip install -r requirements.txt
python src/api_server.py
```

默认监听 `127.0.0.1:8002`，接口文档在 `http://127.0.0.1:8002/docs`。

### 请求

- 方法：`POST`
- 地址：`/v1/inspect`（兼容别名：`/api/v1/inspect`）
- 类型：`multipart/form-data`
- 字段：
  - `image`：必填，图片文件；支持 JPEG、PNG、WEBP、BMP，默认最大 12 MB。
  - `scene`：选填，例如 `起重吊装`、`高处作业`、`施工现场综合安全`。
  - `prompt`：选填，额外检查要求。
  - `detail_crop`：选填，`off`、`auto` 或原图 0–1000 坐标 `x1,y1,x2,y2`；空值遵循服务端配置，显式 `off` 可关闭服务端的局部图设置。非法手工坐标返回 HTTP `400`。
- 若服务端配置了 `API_ACCESS_TOKEN`，请求头必须带 `X-API-Key: <服务访问令牌>`。这不是模型 API Key。

curl 示例：

```bash
curl -X POST "http://127.0.0.1:8002/v1/inspect" \
  -F "image=@/absolute/path/to/frame.jpg" \
  -F "scene=施工现场综合安全" \
  -F "prompt=检查画面中可见的施工安全风险"
```

配置访问令牌时：

```bash
curl -X POST "http://127.0.0.1:8002/v1/inspect" \
  -H "X-API-Key: 替换为服务访问令牌" \
  -F "image=@/absolute/path/to/frame.jpg" \
  -F "scene=施工现场综合安全"
```

需要局部图时，在上述请求增加 `-F 'detail_crop=auto'` 或 `-F 'detail_crop=140,180,790,930'`。仍只运行一次 DeepSeek 图片检测请求，成功响应的 `image_inputs` 为 2。

### 成功响应

HTTP `200`，JSON 字段如下：

```json
{
  "status": "success",
  "format_version": "1.3",
  "output_format": "structured_json",
  "model": "deepseek-flash",
  "scene": "施工现场综合安全",
  "scene_type": "GENERAL_SITE",
  "scene_hint": null,
  "scene_confidence": 0.91,
  "scene_evidence": ["主体设备结构", "设备用途标识"],
  "image": {
    "width": 2000,
    "height": 1000,
    "coordinate_origin": "top_left",
    "coordinate_unit": "pixel"
  },
  "elapsed_seconds": 5.23,
  "issue_count": 1,
  "objects": [],
  "persons": [
    {
      "person_id": "P1",
      "body_visibility": {"head": "complete", "left_hand": "partial", "right_hand": "occluded"},
      "hand_ppe_states": {"left_glove": "uncertain", "right_glove": "uncertain"},
      "ppe_states": {"helmet": "worn", "gloves": "uncertain"},
      "support_surface": "地面",
      "elevation_state": "GROUND_LEVEL"
    }
  ],
  "issues": [
    {
      "item": "吊物下方人员停留",
      "status": "SUSPECTED",
      "risk_category": "LIFTING_OPERATIONS",
      "risk_key": "PERSON_UNDER_SUSPENDED_LOAD",
      "target": "吊物下方人员",
      "position": "图片中的位置",
      "evidence": "图片中可直接观察到的现象",
      "model_regulation_candidate": {
        "standard_name": "模型建议的规范",
        "standard_code": "模型建议编号",
        "article": "模型建议条款",
        "source": "model_internal_knowledge",
        "verification_required": true
      },
      "verified_regulation": {
        "regulation_id": "LIFTING_EXCLUSION_ZONE",
        "standard_name": "建筑与市政施工现场安全卫生与职业健康通用规范",
        "standard_code": "GB 55034-2022",
        "article": "第3.4.1条",
        "clause_text": "吊装作业前应设置安全保护区域……",
        "source": "local_verified_catalog",
        "verification_required": false
      },
      "regulation_match": {
        "method": "risk_key",
        "confidence": 0.95,
        "verified": true
      },
      "regulation": {
        "regulation_id": "LIFTING_EXCLUSION_ZONE",
        "standard_code": "GB 55034-2022",
        "article": "第3.4.1条"
      },
      "rule": "本地已核验的规范名称、编号、条款、条文和来源",
      "confidence": 0.82,
      "bbox_2d_1000": [100, 200, 400, 650],
      "bbox_original_px": {
        "x": 200,
        "y": 200,
        "width": 600,
        "height": 450,
        "image_width": 2000,
        "image_height": 1000
      }
    }
  ],
  "warnings": [],
  "timings": {}
}
```

`bbox_2d_1000` 是模型对整张图估计的归一化 `[x1,y1,x2,y2]`；`bbox_original_px` 是程序换算到 EXIF 方向校正后的原图像素坐标，`x,y` 为左上角，`width,height` 为框宽和框高。无法可靠定位时两者均为 `null`。调用方画框时应使用 `bbox_original_px` 并按 `image.width`、`image.height` 校验边界。

错误响应采用 FastAPI 标准格式，例如 `{"detail":"..."}`。常见 HTTP 状态码为 `400`（空文件或非法裁剪坐标）、`401`（访问令牌错误）、`413`（文件过大）、`415`（不支持的图片格式）、`429`（并发繁忙）、`502`（模型流程失败）、`504`（处理超时）。

**这个对外契约是对接方已经依赖的，不要改成返回检测脚本的原始 JSON。** 检测脚本内部的结构化错误
（`status` / `error.type` / `error.message`）由 API 读取后翻译成 `detail` 文本与状态码：
脚本报 `timeout` → HTTP `504`，其它结构化错误 → HTTP `502`，非零退出 → HTTP `502`，
被信号杀死 → HTTP `502` 并在 `detail` 里写明信号名（如 `SIGKILL`）。

**瞬时故障会自动重试一次**：`timeout` 与 `network_error` 会在总预算内再试一次
（`API_TRANSIENT_RETRIES`，默认 1；`API_TOTAL_TIMEOUT_SECONDS` 默认 17.5 秒）。
`processing_error` 这类真实失败不重试。`REQUEST_TIMEOUT_SECONDS` 是检测脚本自己的预算，
`API_PROCESS_TIMEOUT_SECONDS` 是子进程硬杀上限（默认比检测预算宽 1.5 秒），
调用方等待上限要大于 API 上限。

API 优先读取检测脚本在标准输出中返回的 JSON 错误，Pillow 等依赖写入标准错误的警告不会覆盖真正的失败原因。脚本报告模型超时与 API 子进程超时均返回 `504`；非零退出返回 `502`，信号终止会明确报告信号名称。成功检测不会因标准错误里存在警告而被标记为失败。

线上模型响应超过默认 8 秒时，需要一起调整检测脚本、API 和调用方的等待时间。例如可先按以下设置验证实际响应时间；这会延长等待上限，不保证供应商在预算内完成：

```dotenv
# Flash 项目的 .env
REQUEST_TIMEOUT_SECONDS=30
DEEPSEEK_VISION_TIMEOUT_SECONDS=27
API_PROCESS_TIMEOUT_SECONDS=35
```

使用工作台时，同时在工作台 `.env` 设置 `INSPECT_TIMEOUT_SECONDS=40`，再重启 Flash API 和工作台后端。API 子进程上限应给图片预处理及结果整理留出余量，调用方上限应大于 API 上限。主系统 `L4_STAGE_TIMEOUT_SECONDS` 不参与快速筛查。发生 `SIGKILL` 时，还需核对服务器系统日志是否记录了内存不足等进程终止原因；延长超时不会修复信号终止。

离线验证错误响应及图像质量兼容性：

```bash
python -m unittest discover -s tests -v
```

### 视频端 Python 调用示例

```python
import requests

with open("frame.jpg", "rb") as image_file:
    response = requests.post(
        "http://127.0.0.1:8002/v1/inspect",
        files={"image": ("frame.jpg", image_file, "image/jpeg")},
        data={"scene": "施工现场综合安全"},
        timeout=12,
    )
response.raise_for_status()
result = response.json()
for issue in result["issues"]:
    print(issue["item"], issue["bbox_original_px"])
```

如视频处理程序运行在另一台机器，需要把服务端 `.env` 中 `API_HOST` 设为 `0.0.0.0`，再设置强随机 `API_ACCESS_TOKEN`，并仅在可信网络开放对应端口；不要把模型厂商密钥放入视频端代码。处理视频时建议按场景变化或固定间隔抽帧并限制并发数。接口单次超时由 `API_PROCESS_TIMEOUT_SECONDS` 控制。

## 12. 不同网络下的远程接入

如果视频处理端和检测端不在同一台电脑、也不在同一个局域网，单纯使用 `127.0.0.1` 或 `0.0.0.0` 不够：前者只允许本机访问，后者通常只允许局域网访问。当前项目增加了 `start_remote_api.sh`，用 Cloudflare Tunnel 把本机 HTTP API 映射为 HTTPS 地址，因此双方不需要连接同一个 Wi-Fi，也不要求你配置路由器端口转发。Cloudflare Tunnel 的连接由检测电脑主动向外建立；Quick Tunnel 会生成临时的 `trycloudflare.com` 地址，官方建议只用于开发测试，长期使用应在 Cloudflare 控制台创建命名隧道并绑定自己的域名。citeturn0search3turn0search6turn0search0

### 临时联调

macOS 安装 `cloudflared`：

```bash
brew install cloudflared
```

然后配置服务端 `.env`：

```env
API_HOST=127.0.0.1
API_PORT=8002
API_ACCESS_TOKEN=请替换为至少32位随机字符串
CLOUDFLARE_TUNNEL_MODE=quick
```

启动：

```bash
cd "/Users/gjz/Documents/ChatGPT/复杂环境安全巡检/deepseek_flash_safety_checker"
chmod +x scripts/start_remote_api.sh
./scripts/start_remote_api.sh
```

终端会输出一个类似下面的 HTTPS 地址：

```text
https://random-name.trycloudflare.com
```

学长的视频端调用：

```bash
curl -X POST "https://random-name.trycloudflare.com/v1/inspect" \
  -H "X-API-Key: 你设置的API_ACCESS_TOKEN" \
  -F "image=@/absolute/path/frame.jpg" \
  -F "scene=施工现场综合安全"
```

运行 `start_remote_api.sh` 的终端不能关闭；关闭后隧道地址立即失效。Quick Tunnel 每次启动生成的地址可能变化，适合本次测试，不适合作为固定接口地址。

### 长期使用

在 Cloudflare 控制台创建命名 Tunnel，把公开主机名，例如 `safety-api.example.com`，转发到 `http://127.0.0.1:8002`，复制 Tunnel token 到 `.env`：

```env
CLOUDFLARE_TUNNEL_MODE=named
CLOUDFLARE_TUNNEL_TOKEN=Cloudflare控制台生成的隧道令牌
API_ACCESS_TOKEN=另一组随机服务访问令牌
```

再次运行：

```bash
./start_remote_api.sh
```

学长固定调用：

```text
POST https://safety-api.example.com/v1/inspect
```

这里的 `CLOUDFLARE_TUNNEL_TOKEN` 和 `API_ACCESS_TOKEN` 用途不同：前者让 `cloudflared` 连接你的隧道，后者保护检测 API。两个值都不要发到视频端代码仓库；视频端只保存 API 地址和 `X-API-Key` 服务访问令牌。Cloudflare 官方的命名隧道流程要求在控制台创建 Tunnel、安装并运行 `cloudflared`，再配置公开应用路由到本机服务。citeturn0search0turn0search2
