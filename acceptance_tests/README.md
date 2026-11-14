# 验收测试脚本包

面向第三方测评机构的黑盒测试脚本。只依赖公开 HTTP 接口与公开测试数据，
不导入检测器内部模块，测评方拿到即可执行。

## 1. 依赖

```bash
cd "/Users/gjz/Documents/ChatGPT/复杂环境安全巡检/deepseek_flash_safety_checker"
source .venv/bin/activate
pip install -r requirements.txt
```

视频抽帧另需 `ffmpeg`（`brew install ffmpeg`）或 `opencv-python`，二者其一即可。

## 2. 启动被测服务

```bash
.venv/bin/python src/api_server.py          # 默认 127.0.0.1:8002
```

若服务配置了 `API_ACCESS_TOKEN`，脚本会自动读取同名环境变量作为 `X-API-Key`。
可用 `ACCEPTANCE_API_BASE` 指向其他地址，例如 `http://10.0.0.5:8002`。

## 3. 准备测试数据集

```bash
# 3.1 图库（140 张，带标注）
.venv/bin/python acceptance_tests/build_dataset.py \
    --image-set "/Users/gjz/Documents/ChatGPT/复杂环境安全巡检/评测数据/施工安全图集_140张" \
    --per-scene 100 --out acceptance_tests/dataset

# 3.2 视频抽帧（补足到 10 类场景 × 100 条 = 1000 条）
.venv/bin/python acceptance_tests/extract_frames.py \
    --video /path/to/site.mp4 --scene WORK_AT_HEIGHT \
    --out acceptance_tests/dataset/frames --every 5 --max 120

# 3.3 重新汇总（加入抽帧目录）
.venv/bin/python acceptance_tests/build_dataset.py \
    --image-set "/Users/gjz/Documents/ChatGPT/复杂环境安全巡检/评测数据/施工安全图集_140张" \
    --frames-dir acceptance_tests/dataset/frames \
    --per-scene 100 --out acceptance_tests/dataset
```

`build_dataset.py` 生成的 `dataset_report.json` 会逐场景给出实到数量与缺口。
**样本量不足时脚本明确报缺口，不用重复样本充数。**

## 4. 执行测试

```bash
# 全量
.venv/bin/python acceptance_tests/run_all.py

# 单项
.venv/bin/python acceptance_tests/check_regulation_catalog.py
.venv/bin/python acceptance_tests/test_function_api.py
.venv/bin/python acceptance_tests/test_latency.py --pace 2
```

## 5. 输出物

| 文件 | 内容 |
|---|---|
| `results/catalog/catalog_report.json` | 法规目录核查结论 |
| `results/function/cases.jsonl` | 功能用例逐条执行记录 |
| `results/function/summary.json` | 功能用例汇总 |
| `results/latency/records.jsonl` | 时延逐样本记录 |
| `results/latency/latency_report.json` | 时延统计与 P7 判定 |
| `results/acceptance_summary.json` | 全部测试项汇总 |

## 6. 测试项与验收指标对应

| 脚本 | 大纲测试项 | 指标 |
|---|---|---|
| `test_function_api.py` | FUN-04 智能模型功能测试、FUN-05 预警记录输出 | F1 |
| `test_latency.py` | PER-02 风险预警延迟测试 | P7（≤8 秒） |
| `check_regulation_catalog.py` | DAT-03 场景与风险点覆盖核查 | P3 |

模型准确率、召回率（P4/P5/P6）不在本脚本包范围内，依《项目验收测试大纲 V1.1》
由委托方另行组织评测。
