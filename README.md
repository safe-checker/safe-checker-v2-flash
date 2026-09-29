# DeepSeek Flash 工地安全快速检测

当前版本使用 DeepSeek Flash 对施工现场图片进行单次视觉安全检测。输入为一张图片，可选提供场景提示；输出包含风险描述、可匹配的已核验安全条文，以及每条风险在原图上的范围框坐标。报告同时提供固定字段的中文文本和机器可读 JSON。

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
DEEPSEEK_VISION_TIMEOUT_SECONDS=7.2
MAX_IMAGE_EDGE=768
IMAGE_DETAIL=low
SINGLE_MODEL_MAX_TOKENS=1200
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
python safety_check.py "/图片/绝对路径.jpg" --scene "起重吊装" --dry-run
```

普通运行（默认输出并保存标准 JSON）：

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
报告格式版本：1.0
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

普通运行默认把带缩进的标准 JSON 打印到终端，并保存到 `outputs/` 目录。JSON 不再重复嵌入整段 Markdown，风险、条例和范围框统一放在 `issues` 数组中，其他成员可直接使用 `issues[].bbox_original_px` 绘制范围框。只有需要纯文本报告时才使用 `--text-output`。

## 6. 八秒内响应的关键设置

`.env` 中这些参数会影响速度：

```env
REQUEST_TIMEOUT_SECONDS=8
DEEPSEEK_VISION_TIMEOUT_SECONDS=7.2
IMAGE_DETAIL=low
MAX_IMAGE_EDGE=768
JPEG_QUALITY=70
SINGLE_MODEL_MAX_TOKENS=1200
TEMPERATURE=0
REASONING_EFFORT=none
```

如果经常超过 8 秒，优先调小：

```env
MAX_IMAGE_EDGE=640
SINGLE_MODEL_MAX_TOKENS=900
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

### 7.1 真实、可查询的法规条款

视觉模型负责识别图片中的对象、状态、风险和证据，但不直接生成最终法规文本。模型只能返回法规目录中的 `citation_key`，程序再从 `verified_regulations.py` 读取标准名称、条款号、条文原文和查询链接。模型返回未知键或无法对应时，报告会明确写“暂未匹配到已核验的具体条文”，不会把模型自由生成的标准编号当作真实条例。

当前临时用电目录使用现行《建筑与市政工程施工现场临时用电安全技术标准》JGJ/T 46-2024。查询入口为中国建筑科学研究院建筑防火研究所标准页面：https://gf.cabr-fire.com/article-68828.htm。报告中的条款仍应结合项目所在地、工程日期和适用标准版本由安全专业人员确认。

当前单模型检测链路：

1. DeepSeek Flash 完成图片识别和施工安全判断，代码不再调用其他视觉或文本模型。
2. Prompt 要求说明对象、位置、可见证据、风险状态和 0–1000 归一化边界框；未能可靠定位时返回 `null`。
3. 代码只做通用字段校验、合法法规键校验、坐标裁剪及从归一化坐标到原图像素坐标的换算，不按场景关键词创建或覆盖风险结论。
4. 违规范围框输出为原图像素坐标：左上角 `x,y`，以及 `width,height`；图像按 EXIF 方向校正后确定坐标系。
5. 法规引用由 `verified_regulations.py` 白名单补全。当前已核验目录集中在施工现场临时用电；目录没有对应条款时报告会明确写“暂未匹配到已核验的具体条文”，不让模型编造法规。

因此，图片受强光、反光、遮挡、模糊影响时，结果会更倾向于输出“疑似违规”或“不可判断”，减少把不确定内容写成确定违规。

8 秒控制方式：

```text
总超时 REQUEST_TIMEOUT_SECONDS=8
DeepSeek Flash 单次视觉请求 DEEPSEEK_VISION_TIMEOUT_SECONDS=7.2
剩余时间用于 JSON 解析、坐标换算和报告输出
```

单模型减少了多路并发等待，但第三方 API、网络和服务端排队仍可能导致总时长超过 8 秒；该时间要求是目标，不是任何网络情况下的绝对保证。

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

项目提供 `api_server.py` HTTP API，视频处理端可上传图片并获取 DeepSeek Flash 单次检测结果及原图坐标框。视频端不需要、也不应传递模型厂商的 API Key。

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
  "format_version": "1.0",
  "output_format": "structured_json",
  "model": "deepseek-flash",
  "scene": "施工现场综合安全",
  "scene_type": "GENERAL_SITE",
  "image": {
    "width": 2000,
    "height": 1000,
    "coordinate_origin": "top_left",
    "coordinate_unit": "pixel"
  },
  "elapsed_seconds": 5.23,
  "issue_count": 1,
  "issues": [
    {
      "item": "风险名称",
      "status": "SUSPECTED",
      "target": "风险对象",
      "position": "图片中的位置",
      "evidence": "图片中可直接观察到的现象",
      "rule": "已核验规范标准名称、编号、条款原文和查询链接，或未匹配提示",
      "confidence": 0.82,
      "citation_key": "UNKNOWN",
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
