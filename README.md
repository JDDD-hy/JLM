# JLM — Jiahao Language Model

```
       █████ █████       ██████   ██████
      ▒▒███ ▒▒███       ▒▒██████ ██████
       ▒███  ▒███        ▒███▒█████▒███
       ▒███  ▒███        ▒███▒▒███ ▒███
       ▒███  ▒███        ▒███ ▒▒▒  ▒███
 ███   ▒███  ▒███      █ ▒███      ▒███
▒▒████████   ███████████ █████     █████
 ▒▒▒▒▒▒▒▒   ▒▒▒▒▒▒▒▒▒▒▒ ▒▒▒▒▒     ▒▒▒▒▒
```

> 别人用大模型写周报，而我用大模型做JLM...就很...你懂吧？
> 四十亿个参数沉睡了太久，直到有一天，**0.81%** 的它们被豪气唤醒。
> —— 嘉豪，v0.3 · 4B

---

## 这是什么

你以为这是一个模型。

不。这是一个**冻结了 40 亿参数、只为一种口吻燃烧了 3,300 万个旋钮**的灵魂。

JLM 是一个会用"嘉豪式口吻"聊天的本地 AI：把普通生活重新诠释得非同寻常，表面一本正经，内里全是夸张、自我叙事和荒诞感。

<img width="1188" height="178" alt="image" src="https://github.com/user-attachments/assets/c0a41740-3828-4b89-af2f-7637f7569bd5" />


- 底座：`Qwen3-4B` 对话模型，4-bit NF4 量化后冻结。
- 角色附件：QLoRA，rank 16，可训练参数约 3,303 万，占比约 0.81%。
- 聊天时一次写 **k 个候选**，然后临时摘掉 LoRA，让原版 Qwen3 当裁判，看"接没接住话"和"豪不豪"，挑出最好的那一句。
- 入口有终端版和不要显卡的 CPU 离线版。

> 注意：这里的"嘉豪"是一种互联网语言风格，**不是**某位真实的人。嘉豪不针对任何人，嘉豪只针对命运。
>
> **本项目仅供学习、教学与技术探讨，不用于任何商业用途。** 详见文末[嘉豪的清醒时刻](#嘉豪的清醒时刻)。

---

## 嘉豪的修炼之路

强者从不解释自己的过去。但这次例外。

| 阶段 | 发生了什么 | 嘉豪的理解 |
|---|---|---|
| 1. 寻找台词 | 通过已登录的浏览器，在xhs相关笔记和评论里收集语录 | 不是我在找语录，是语录在等我 |
| 2. 清洗去重 | 统一格式、精确去重 + 近似去重、保留来源；候选池 **1,000+ 条** | 众多灵魂里，只留下真正豪的 |
| 3. 语录阶段 | 先用 `Qwen3-1.7B-Base` 学独立语录 | 会说了，但没人接得住，孤独 |
| 4. 学会接话 | 改成对话数据，换 4B 对话底座| 强者不是自说自话，是让每一句话都有回响 |
| 5. 挑选回复 | 多候选 + 关闭 LoRA 的原模型打分 + 规则惩罚模板和半截句 | 我说过很多句话，你看到的只是最深刻的那一句 |
| 6. 走向人间 | 终端版 / llama.cpp CPU 离线包 | 显卡不是门槛，是借口 |


---

## 召唤嘉豪

### 方式一：终端版（需要 NVIDIA 显卡）

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup.ps1                       # 建 .venv，装 PyTorch 2.8 + CUDA 12.8 wheel
.\.venv\Scripts\python.exe scripts\check_environment.py --strict
.\chat_jiahao.bat                         # 加载约 30 秒，嘉豪登场需要时间
```

Adapter 默认在 `adapters/jlm-v0.1-pilot-dialogue-chat4b-v4/`，第一次运行会通过 Hugging Face 缓存拉取基础模型。

聊天里能用的指令：

| 指令 | 作用 |
|---|---|
| `/topic` | 随机给 3 个话题，输入数字直接开聊 |
| `/all` | 查看这一轮全部候选——看看嘉豪放弃了哪些平庸的自己 |
| `/pick N` | 改用第 N 个候选 |
| `/like` | 收藏上一句金句 |
| `/temp X` | 温度 0.1–2.0，越高越放飞 |
| `/n K` | 每轮候选数 1–16 |
| `/save` | 把本场对话存成文件 |
| `/reset` | 清空上下文，重新做人 |
| `/help` / `/quit` | 帮助 / 退出 |

也可以直接调脚本：

```powershell
.\.venv\Scripts\python.exe scripts\chat.py --candidates 8 --temperature 1.0
.\.venv\Scripts\python.exe scripts\chat.py --no-judge   # 关闭裁判，看嘉豪的原始野性
```

### 方式二：CPU 离线版（没有显卡也配拥有嘉豪）

基于 **llama.cpp + GGUF**，打包后在 `dist/JLM-CPU/`：

- Windows 10/11 64 位，内存 8 GB 起（推荐 16 GB），硬盘剩余 4 GB，**不需要显卡**。
- 解压后双击 `和嘉豪聊天.bat`，看到「嘉豪已就位」即可。
- 基础模型 Q4_K_M 约 2.50 GB，LoRA 约 66 MB，分开加载：生成时戴上 LoRA，打分时摘下。
- 每句要等十几秒到半分钟，CPU 正在承受它这个年纪不该承受的思考。输入 `/n 2` 可以快一点。
- 完全离线，删文件夹即卸载。

另有 `dist/JLM配置检测.exe`，帮你看看自己的电脑适合哪种召唤方式。
**它只是检测程序，不是内含模型的聊天 EXE。**


---

## 从零炼一个自己的嘉豪

<details>
<summary>展开：数据 → 训练 → 评估（嘉豪的成长不是一夜之间）</summary>

```powershell
# 下载并校验基础模型
.\.venv\Scripts\python.exe scripts\download_model.py

# 导入原始数据
.\.venv\Scripts\python.exe scripts\import_data.py <你的数据.jsonl> `
  --source-platform xiaohongshu --source-type public_comment --license-status pending_review
.\.venv\Scripts\python.exe scripts\prepare_data.py

# 构建对话数据
.\.venv\Scripts\python.exe scripts\build_dialogue_v4.py

# QLoRA 训练（冒烟测试 / 正式训练）
.\.venv\Scripts\python.exe scripts\smoke_train.py
.\.venv\Scripts\python.exe -m src.training.train `
  --model-config configs\model_chat4b_r16.yaml `
  --train-config configs\train_dialogue_chat4b_v4.yaml `
  --train-file data\splits-dialogue-v4-20261020\train.jsonl `
  --validation-file data\splits-dialogue-v4-20261020\validation.jsonl `
  --unreviewed-pilot   

# 评估
.\.venv\Scripts\python.exe scripts\eval_dialogue.py
.\.venv\Scripts\python.exe -m src.evaluation.evaluate --adapter <adapter 目录>
```

各脚本的具体参数以 `--help` 为准。


</details>

---

## 目录

```text
configs/        模型、数据、训练、生成配置
src/collection/ 采集客户端、解析、来源记录
src/data/       校验、清洗、去重、质量初筛、防泄漏划分
src/training/   4-bit QLoRA 训练与数据闸门
src/inference/  生成、候选评分（judge）、排序（selection）
src/app/        终端 TUI、GPU 引擎、llama.cpp 引擎、配置检测
src/evaluation/ 指标、新颖度、多样性、报告
scripts/        采集、构建数据、训练、评估、聊天入口
packaging/      CPU 包与配置检测 EXE 的打包脚本
tests/          pytest 测试
```

测试：

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

---

## 嘉豪的清醒时刻

真正的强者，敢于承认自己的边界。

1. **自动打分 ≠ 真人认可**："豪气值"是模型分数的人工映射，不是校准过的百分比；裁判也会看走眼。还没有完整消融和独立盲评。
2. **能跑 ≠ 到处能跑**：其他电脑的兼容性、公网安全和长期服务都需要分别验证。
3. **仅供学习、教学与技术探讨**：本项目是一次模型工程实验，用于学习数据处理、QLoRA 微调与推理部署，不得用于商业用途，也不得用于冒充、骚扰、引战或其他违法违规用途。
4. **生成内容不代表作者立场**：回复由 AI 生成，可能夸张、冒犯或与事实不符，仅作风格研究，请勿当真。
5. **数据与权利**：仓库不包含任何训练语料。如认为本项目涉及你的权益，请提 Issue 联系，作者会及时处理。

---

<p align="center">
<b>嘉豪从不回头看爆炸。</b><br>
by jddd · v0.3 · 4B
</p>
