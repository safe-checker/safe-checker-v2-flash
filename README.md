# 多模型复杂工地安全快速检测

这个小项目用于通过 DeepSeek Flash、Qwen、GLM、Kimi 和豆包五个视觉模型对施工现场图片做一次性多场景安全快速检测。五个模型使用同一套安全检测 Prompt，平权并行读取同一张图片，再通过置信度评分和投票融合生成最终风险初筛报告。未配置某个模型的 API Key 时，程序会自动跳过该模型并记录提示。

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

## 1. 环境要求

- Python 3.10 或更高版本
- VS Code
- DeepSeek API Key
- 通义千问 API Key
- GLM API Key
- Kimi API Key
- 火山方舟豆包 API Key 和视觉模型推理接入点 ID

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
QWEN_API_KEY=sk-你的通义千问真实密钥
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
QWEN_REVIEW_MODEL=qwen3.8-flash
ENABLE_QWEN_VISION_REVIEW=true
QWEN_VISION_MODEL=qwen3.8-flash
ENABLE_THIRD_VISION_REVIEW=true
THIRD_VISION_BASE_URL=https://open.bigmodel.cn/api/paas/v4/chat/completions
THIRD_VISION_MODEL=glm-5.3-flash
ENABLE_STRONG_REVIEW=true
ENABLE_KIMI_VISION=true
KIMI_API_KEY=sk-你的Kimi真实密钥
KIMI_BASE_URL=https://api.moonshot.cn/v1/chat/completions
KIMI_VISION_MODEL=kimi-k2.6
ENABLE_DOUBAO_VISION=true
DOUBAO_API_KEY=你的火山方舟真实密钥
DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3/chat/completions
DOUBAO_VISION_MODEL=ep-你的豆包视觉接入点ID
REQUEST_TIMEOUT_SECONDS=8
PRIMARY_TIMEOUT_SECONDS=4.2
REVIEW_TIMEOUT_SECONDS=3.2
QWEN_VISION_TIMEOUT_SECONDS=6.8
KIMI_VISION_TIMEOUT_SECONDS=7.2
DOUBAO_VISION_TIMEOUT_SECONDS=7.2
```

`QWEN_API_KEY` 用于 Qwen 平权视觉检测。若暂时不填写，程序会跳过 Qwen，并在报告中说明实际参与投票的模型数量。

GLM、Kimi 和豆包都作为平权视觉模型参与检测。拿到对应 API Key 后，填写：

```env
ENABLE_THIRD_VISION_REVIEW=true
THIRD_VISION_API_KEY=sk-你的第三模型密钥
THIRD_VISION_MODEL=glm-5.3-flash
```

GLM、Kimi 和豆包与 DeepSeek、Qwen 一样都是平权视觉模型。使用不同厂商/不同架构的模型，有助于减少同源偏差；当前配置最多可由五个模型共同参与投票。

Kimi 和豆包会与 DeepSeek、Qwen、GLM 平权并行进行图片安全检测。Kimi 使用视觉模型名；豆包通常使用火山方舟控制台创建的视觉模型推理接入点 ID，例如 `ep-xxxxxxxx-xxxxx`：

```env
KIMI_API_KEY=你的KimiAPI密钥
KIMI_VISION_MODEL=kimi-k2.6
DOUBAO_API_KEY=你的火山方舟API密钥
DOUBAO_VISION_MODEL=ep-你的豆包视觉接入点ID
```

如果你的 DeepSeek Flash API 地址或模型名和默认值不同，请按实际平台文档修改：

```env
DEEPSEEK_BASE_URL=https://api.deepseek.com/chat/completions
DEEPSEEK_MODEL=deepseek-flash
```

Qwen 默认使用阿里云百炼 OpenAI 兼容接口。当前实验中，`qwen3.8-flash` 同时承担图片二次复检；如果更看重视觉复检速度，也可以切换回 `qwen-vl-plus`：

```env
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
QWEN_REVIEW_MODEL=qwen3.8-flash
QWEN_VISION_MODEL=qwen3.8-flash
```

## 4. 运行检测

先用 dry-run 验证图片和配置，不会调用 API：

```bash
python safety_check.py "/图片/绝对路径.jpg" --scene "起重吊装" --dry-run
```

普通运行：

```bash
python safety_check.py "/图片/绝对路径.jpg"
```

如果已知场景，建议加上 `--scene`，这样更快、更准：

```bash
python safety_check.py "/图片/绝对路径.jpg" --scene "起重吊装"
```

常见场景示例：

```bash
python safety_check.py "/图片/绝对路径.jpg" --scene "施工现场临时用电"
python safety_check.py "/图片/绝对路径.jpg" --scene "高处作业"
python safety_check.py "/图片/绝对路径.jpg" --scene "基坑工程"
python safety_check.py "/图片/绝对路径.jpg" --scene "吊装作业"
python safety_check.py "/图片/绝对路径.jpg" --scene "塔吊"
python safety_check.py "/图片/绝对路径.jpg" --scene "施工升降机"
python safety_check.py "/图片/绝对路径.jpg" --scene "脚手架"
```

## 5. 输出格式

终端会输出：

```text
耗时：6.23 秒

基于图片中清晰可见的施工现场情况，识别出以下安全违规点及对应的安全规范：

1. 违规点：……
• 现象描述：……
• 违反的具体安全条例：《建筑与市政工程施工现场临时用电安全技术标准》JGJ/T 46-2024 第……条：……
  查询链接：https://gf.cabr-fire.com/article-68828.htm
```

同时会把结果保存到 `outputs/` 目录。

## 6. 八秒内响应的关键设置

`.env` 中这些参数会影响速度：

```env
REQUEST_TIMEOUT_SECONDS=8
IMAGE_DETAIL=low
MAX_IMAGE_EDGE=768
JPEG_QUALITY=70
MAX_TOKENS=1100
REVIEW_MAX_TOKENS=350
QWEN_MAX_IMAGE_EDGE=448
QWEN_JPEG_QUALITY=50
QWEN_VISION_MAX_TOKENS=350
THIRD_MAX_IMAGE_EDGE=256
THIRD_JPEG_QUALITY=35
THIRD_VISION_MAX_TOKENS=700
KIMI_MAX_IMAGE_EDGE=384
KIMI_JPEG_QUALITY=45
KIMI_VISION_MAX_TOKENS=320
DOUBAO_MAX_IMAGE_EDGE=256
DOUBAO_JPEG_QUALITY=38
DOUBAO_VISION_MAX_TOKENS=448
TEMPERATURE=0
REASONING_EFFORT=none
```

如果经常超过 8 秒，优先调小：

```env
MAX_IMAGE_EDGE=768
MAX_TOKENS=600
```

注意：`high` 和 `max` 会自动提高实际请求的 `max_tokens`，为思考过程和最终报告预留空间，因此这两种模式可能超过 8 秒。

深度思考强度可以这样设置：

```env
REASONING_EFFORT=none   # 关闭思考，最快
REASONING_EFFORT=none   # 关闭思考，优先保证快速返回
REASONING_EFFORT=high   # 更充分思考，速度较慢
REASONING_EFFORT=max    # 最大思考，速度最慢
```

当前任务如果要求 8 秒内稳定返回，建议使用：

```env
REASONING_EFFORT=none
```

原因是增强 Prompt 已经加入了证据约束和误判抑制，继续使用 `low/high` 可能把输出 token 消耗在思考内容上，导致超时或没有最终报告。

## 7. 误判抑制机制

### 7.1 真实、可查询的法规条款

视觉模型负责识别图片中的对象、状态、风险和证据，但不直接生成最终法规文本。模型只能返回法规目录中的 `citation_key`，程序再从 `verified_regulations.py` 读取标准名称、条款号、条文原文和查询链接。模型返回未知键或无法对应时，报告会明确写“暂未匹配到已核验的具体条文”，不会把模型自由生成的标准编号当作真实条例。

当前临时用电目录使用现行《建筑与市政工程施工现场临时用电安全技术标准》JGJ/T 46-2024。查询入口为中国建筑科学研究院建筑防火研究所标准页面：https://gf.cabr-fire.com/article-68828.htm。报告中的条款仍应结合项目所在地、工程日期和适用标准版本由安全专业人员确认。

当前版本针对复杂工地安全场景增加了多层误判抑制：

1. 五个视觉模型平权并行检测：DeepSeek Flash、Qwen、GLM、Kimi、豆包（按配置启用情况参与）同时读取同一张图片，分别完成图片识别和安全风险判断；不存在主模型替其他模型裁决的层级关系。
2. 代码只做编排和质量控制：负责图片压缩、超时控制、JSON 解析、五模型投票融合、证据字段校验和报告格式化，不把有限规则库作为主要裁判。
3. 风险归并：优先使用模型输出的稳定 `risk_key`，同时检查目标对象、位置和文本证据是否相容。相同法规编号不能单独证明是同一风险，因此同一法规下不同对象或不同位置的风险会分别保留。
4. 低票风险保留：多数票风险优先展示；只有一个或少数模型发现、但证据仍可描述的风险也会保留，并标记为“少数模型发现，待人工复核”，不会因为票数少而静默删除。
5. 置信度融合：`model_evidence_confidence` 表示支持模型的证据质量均值，`consensus_score` 表示支持模型占已完成模型的比例，`configured_support_rate` 表示支持模型占已配置模型的比例，`model_availability_rate` 表示已完成模型占已配置模型的比例；`vote_confidence` 综合证据、共识、配置覆盖率，并对未完成模型进行可用率惩罚，单模型自信不能直接变成高融合置信度。
6. 实名投票明细：每条风险都会逐一列出 DeepSeek、Qwen、GLM、Kimi、豆包的投票状态。`支持` 表示该模型返回了同一根本风险，`未支持` 表示该模型已完成图片检测但没有提出该风险，`未返回` 表示调用失败、超时或 JSON 无法解析。
7. 事实字段辅助复核：`yes/no/uncertain/not_applicable` 用来判断对象是否存在、状态是否可见，但不限制模型发现其他安全风险。
8. Qwen 文本复核兜底：当视觉复检失败且事实字段存在逻辑冲突时，只复核 JSON 一致性。
9. 通用证据门控：每个风险必须同时说明 `target_visibility` 和 `evidence_level`；关键部位未入镜、被遮挡或证据不直接时，代码只做通用证据质量控制，不替模型新增风险。
10. 文本复核模型只检查 JSON 是否自相矛盾，复核意见单独记录为人工复核提示，不覆盖主模型视觉事实。
11. 代码不再把“箱门打开”强制改写成“插座口外露”或直接改成“不可判断”；配电箱主箱门、插座盖、插头、接线端子和电缆接口由视觉模型根据实际形态分别识别。

因此，图片受强光、反光、遮挡、模糊影响时，结果会更倾向于输出“疑似违规”或“不可判断”，减少把不确定内容写成确定违规。

8 秒控制方式：

```text
总超时 REQUEST_TIMEOUT_SECONDS=8
Flash 初筛 PRIMARY_TIMEOUT_SECONDS=4.2
Qwen 图像复检 QWEN_VISION_TIMEOUT_SECONDS=7.2
第三模型复检 THIRD_VISION_TIMEOUT_SECONDS=7.2
Kimi 图片检测 KIMI_VISION_TIMEOUT_SECONDS=7.2
豆包图片检测 DOUBAO_VISION_TIMEOUT_SECONDS=7.2
Qwen 文本复核 REVIEW_TIMEOUT_SECONDS=3.2
剩余时间用于规则汇总和输出
```

各视觉模型是并行进行的，所以总耗时接近最慢的一路模型，而不是三次相加。如果某一路复检模型没有在时间预算内返回，程序会保留已完成模型的结果，并在报告中提示该模型未完成。

## 8.1 高精度模型检查协议

当前 Prompt 要求视觉模型按以下顺序完成一次检查：

```text
对象清单
→ 人员及关键部位完整性
→ 空间关系和作业位置
→ 多类别安全风险扫描
→ 规范匹配
→ 证据自检
```

每个模型风险项都要返回：

```json
{
  "target": "风险对象或人员编号",
  "target_visibility": "complete/partial/occluded/not_visible",
  "evidence_level": "direct/partial/occluded/not_visible/ambiguous",
  "citation_confidence": "high/medium/low/unknown"
}
```

例如，人员头部没有完整进入画面时，模型不能写“未戴安全帽”，而应写“头部未完整入镜，无法判断安全帽状态”。代码只依据模型提供的可见性和证据等级进行降级，因此这套机制可以同时适用于安全帽、反光衣、安全带、防护栏、接地、防脱装置等不同风险，不需要为每个物体单独编写违规判断规则。

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
- 8 秒内完成依赖网络、图片大小和模型服务状态；脚本会强制设置 8 秒请求超时，但不能保证第三方服务一定在 8 秒内返回。
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

项目原先只有命令行入口，没有供其他程序直接连接的 HTTP API。现在新增 `api_server.py`，视频处理端可逐帧上传图片，服务端继续使用本机 `.env` 中配置的五个视觉模型并行检测。视频端不需要、也不应传递任何模型厂商的 API Key。

安装依赖并启动服务：

```bash
cd "/Users/gjz/Documents/ChatGPT/复杂环境安全巡检/deepseek_flash_safety_checker"
source .venv/bin/activate
pip install -r requirements.txt
python api_server.py
```

默认监听 `127.0.0.1:8000`，接口文档在 `http://127.0.0.1:8000/docs`。

### 请求

- 方法：`POST`
- 地址：`/v1/inspect`（兼容别名：`/api/v1/inspect`）
- 类型：`multipart/form-data`
- 字段：
  - `image`：必填，图片文件；支持 JPEG、PNG、WEBP、BMP，默认最大 12 MB。
  - `scene`：选填，例如 `起重吊装`、`高处作业`、`施工现场综合安全`。
  - `prompt`：选填，额外检查要求。
- 若服务端配置了 `API_ACCESS_TOKEN`，请求头必须带 `X-API-Key: <服务访问令牌>`。这不是模型 API Key。

curl 示例：

```bash
curl -X POST "http://127.0.0.1:8000/v1/inspect" \
  -F "image=@/absolute/path/to/frame.jpg" \
  -F "scene=施工现场综合安全" \
  -F "prompt=检查画面中可见的施工安全风险"
```

配置访问令牌时：

```bash
curl -X POST "http://127.0.0.1:8000/v1/inspect" \
  -H "X-API-Key: 替换为服务访问令牌" \
  -F "image=@/absolute/path/to/frame.jpg" \
  -F "scene=施工现场综合安全"
```

### 成功响应

HTTP `200`，JSON 字段如下：

```json
{
  "status": "success",
  "scene": "施工现场综合安全",
  "scene_type": "GENERAL_SITE",
  "elapsed_seconds": 6.23,
  "issues": [
    {
      "item": "风险名称",
      "status": "SUSPECTED",
      "vote_status": "SUSPECTED",
      "evidence": "图片中可直接观察到的现象",
      "rule": "对应的已核验规范，或暂未匹配到已核验的具体条文",
      "vote_support": 2,
      "vote_total": 5,
      "named_votes": [
        {"display_name": "DeepSeek Flash", "vote": "支持"},
        {"display_name": "Qwen3.8-Flash", "vote": "未支持"}
      ]
    }
  ],
  "vision_review": {},
  "warnings": [],
  "timings": {},
  "report": "便于人阅读的完整检测报告文本"
}
```

`issues` 中每项还会包含模型证据均值、共识度、配置覆盖率、模型完成率、融合置信度、证据等级和法规键等字段；`named_votes` 为实际参与模型的逐模型明细。`未支持` 的含义是该模型完成了检测但没有返回这个风险，不等于它明确判断现场安全；`未返回` 表示模型调用失败或超时。少数模型发现的候选会保留并标记待复核，业务端应同时保留 `report` 和完整 JSON，不要只用投票数过滤风险。

错误响应采用 FastAPI 标准格式，例如 `{"detail":"..."}`；常见 HTTP 状态码为 `400`（空文件）、`401`（访问令牌错误）、`413`（文件过大）、`415`（不支持的图片格式）、`429`（并发繁忙）、`502`（模型流程失败）、`504`（处理超时）。

### 视频端 Python 调用示例

```python
import requests

with open("frame.jpg", "rb") as image_file:
    response = requests.post(
        "http://127.0.0.1:8000/v1/inspect",
        files={"image": ("frame.jpg", image_file, "image/jpeg")},
        data={"scene": "施工现场综合安全"},
        timeout=12,
    )
response.raise_for_status()
result = response.json()
for issue in result["issues"]:
    print(issue["item"], issue["vote_support"], "/", issue["vote_total"])
```

如视频处理程序运行在另一台机器，需要把服务端 `.env` 中 `API_HOST` 设为 `0.0.0.0`，再设置强随机 `API_ACCESS_TOKEN`，并仅在可信内网开放对应端口；不要把模型厂商密钥放入视频端代码。处理视频时建议先按场景变化或固定时间间隔抽帧，限制并发数，避免每一帧都同时触发五个模型造成费用和排队延迟。接口单次超时由 `API_PROCESS_TIMEOUT_SECONDS` 控制，第三方模型延迟仍可能使请求超时。

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
API_PORT=8000
API_ACCESS_TOKEN=请替换为至少32位随机字符串
CLOUDFLARE_TUNNEL_MODE=quick
```

启动：

```bash
cd "/Users/gjz/Documents/ChatGPT/复杂环境安全巡检/deepseek_flash_safety_checker"
chmod +x start_remote_api.sh
./start_remote_api.sh
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

在 Cloudflare 控制台创建命名 Tunnel，把公开主机名，例如 `safety-api.example.com`，转发到 `http://127.0.0.1:8000`，复制 Tunnel token 到 `.env`：

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
