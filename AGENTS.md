# AGENTS.md — safe-checker-v2-flash

Codex 在本目录启动时会自动读这份文件。

## 这是什么

单张施工现场照片 → DeepSeek Flash **一次**视觉调用完成识图与安全判断 → Python 用本地已核验法规目录
核验条款 → 输出标准 JSON（含原图坐标框 `x,y,width,height`）+ Markdown 报告。

## 目录分工

| 目录 | 放什么 |
|---|---|
| `src/` | 运行时代码（手写）；`src/catalogs/` 是**自动生成**的，勿手改 |
| `data/` | 规则与法规数据，只读输入 |
| `tools/` | 生成与维护脚本；改数据后跑这里 |
| `tests/` | 离线测试（101 项，不调模型） |
| `scripts/` | 远程部署、图像输入对比、组会 PPT |
| `docs/` | 流程说明与实验记录 |

## 命令

```bash
.venv/bin/python src/safety_check.py "/图片/绝对路径.jpg" --scene "起重吊装"
.venv/bin/python src/api_server.py                     # HTTP API，端口 8002
./scripts/start_remote_api.sh                         # Cloudflare Tunnel
.venv/bin/python -m unittest discover -s tests -v     # 101 项离线测试
```

## 改数据的正确顺序

1. 改 `data/` 下的输入，或改 `tools/` 下的生成脚本；
2. 重跑对应的生成脚本（`build_rulepack_catalog.py` / `build_scene_checklist.py` / `build_standard_catalog.py`）；
3. 跑 `python -m unittest discover -s tests -v`；
4. 提交 `data/`、`tools/`、`src/catalogs/` 三处。

**生成必须可复现**：同样输入重跑一次，产出逐字节相同。
`src/catalogs/` 里的触发器 ID 用稳定摘要，不要用 `hash()`（Python 每进程随机化种子）。

## 约定

- 法规目录只做事后核验，**绝不进提示词**；模型只写「标准编号+条号」。
- 被 import 的模块里任何输出必须走 **stderr**，打到 stdout 会污染检测器的 JSON 输出。
- 跑测期间不要改动 `src/` 或 `data/` 下的文件，子进程会 import 到写了一半的模块。
- 报告里的数字必须来自真实跑测记录，不得编造。

## 评估：必须用 LLM 裁判

```bash
.venv/bin/python tools/run_scope_ab.py /tmp/safety_eval/results/scope_ab 5   # 跑 A/B
.venv/bin/python tools/judge_detections.py \
    /tmp/safety_eval/results/scope_ab/records.jsonl \
    /tmp/safety_eval/图片集_140张/labels.jsonl                            # 裁判
.venv/bin/python tools/score_arms.py /tmp/safety_eval/results/scope_ab     # 汇总
```

词法匹配会把中文改述全判成未命中，**不要用它下结论**。
跑测记录里必须保留每路的错误类型（`branch_errors`）：
一批因为限流而全错的记录，看起来和"什么都没检出来"一模一样，必须能分开。

## 已知问题（2026-11-02）

- `SCENE_CHECKLIST_SCOPE` 默认是 `family`：规则包那 62 个主体按
  脚手架/起重/临电/基坑·高处·综合四族切给 4 路并发，四路并起来覆盖全部。
  同一批 35 张上命中 77.1%（旧版场景锁死切分只有 40.0%，加场景专属主体只有 31.4%）。
  **换成 `split` 或 `full` 之前先想清楚：场景专属主体的措辞是"配置齐不齐"，
  标注问的是"缺陷发没发生"，层面不对，加进去反而掉分。**
- `RULE_PACK_ONLY=1`（默认开）把 `MS-`/`GEN-` 前缀的主系统主体排除在清单外；
  `CONCISE_SUBJECTS=1`（默认开）把每个主体精简成前 4 个维度、前 3 句违规描述。
  这两个开关一起决定清单长度，改任何一个都要重新量提示词体积和耗时。
- `.env` 在仓库根，`safety_check.py` 在 `src/`。代码用 `_find_env_file()` 向上找，
  别改回同目录查找——那会让命令行运行全部报 `missing_api_key`。
- `SINGLE_MODEL_MAX_TOKENS=2200` 对当前 schema 偏小，输出会被截断。
  `_salvage_partial_issue()` 能把写了一半的那条风险救回来。
