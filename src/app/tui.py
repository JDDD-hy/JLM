"""Rich terminal UI for chatting with 嘉豪 (Jiahao Language Model)."""

from __future__ import annotations

import json
import random
import statistics
import threading
import time
from datetime import datetime
from pathlib import Path

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.app.branding import AUTHOR, BANNER_COLORS, GOLD, SHADE_COLOR, SUBTITLE, TITLE_ART, VERSION, prepare_console
from src.app.engine import JiahaoEngine

# Splash greeting: 嘉豪 introducing himself, so the opening line reads as a hello rather than a random quip.
QUOTES = [
    "初次见面，我叫嘉豪。不用记住我，你会忘不掉的。",
    "从小到大没人敢直呼我全名，都是恭恭敬敬喊我「豪哥」。",
    "你们都说我是嘉豪，我是豪，但没有家。",
    "自我介绍一下：嘉豪，爱好控分，特长是让参考答案避我三分锋芒。",
    "别人自我介绍报名字，我自我介绍报战绩。",
    "我叫嘉豪，名字里带个豪，是怕你们认不出我。",
    "我是嘉豪。别人有的我都有，别人没有的，我出门顺手买了。",
    "你好，我是嘉豪。今天心情不错，允许你和我说话。",
    "我是嘉豪，一个被同龄人追赶了二十年的普通人。",
    "叫我嘉豪就行，豪哥是别人对我的尊称。",
]
LOADING_LINES = [
    "正在召唤嘉豪……",
    "嘉豪正在整理发型……",
    "嘉豪正在给参考答案上课……",
    "嘉豪正在控分，请稍候……",
    "嘉豪正在思考要不要理你……",
]
THINKING_LINES = ["嘉豪正在酝酿……", "嘉豪在组织措辞……", "嘉豪微微一笑……", "嘉豪在控制自己的实力……"]
TOPICS = [
    "你吃饭了吗",
    "你这次考试考得怎么样",
    "周末一起去打球吗",
    "借我点钱呗",
    "你会游泳吗",
    "你王者什么段位",
    "你有对象吗",
    "你英语怎么样",
    "你昨晚几点睡的",
    "能借我抄下作业吗",
    "你会做饭吗",
    "你跑八百米多少秒",
    "实习找到了吗",
    "你是不是有点装",
    "今天天气好热啊",
    "老师让你去办公室一趟",
    "你会写代码吗",
    "你过年回家吗",
    "推荐本书呗",
    "你健身吗",
]
LEVELS = [(0.4, "见习嘉豪"), (0.6, "嘉豪学徒"), (0.75, "嘉豪"), (0.9, "豪哥"), (1.01, "嘉豪本豪")]
HELP_ROWS = [
    ("/topic", "随机给 3 个话题，输入数字直接开聊"),
    ("/all", "查看这一轮全部候选回复"),
    ("/pick N", "改用第 N 个候选"),
    ("/like", "收藏上一句金句"),
    ("/temp X", "调整温度 0.1–2.0（越高越放飞）"),
    ("/n K", "每轮候选数 1–16"),
    ("/save", "把本场对话存成文件"),
    ("/reset", "清空上下文，重新开始"),
    ("/help", "显示这张表"),
    ("/quit", "退出"),
]


def bar(value: float, width: int = 10) -> str:
    filled = max(0, min(width, round(value * width)))
    return "█" * filled + "░" * (width - filled)


def level_of(score: float) -> str:
    return next(name for limit, name in LEVELS if score < limit)


class JiahaoApp:
    def __init__(self, engine: JiahaoEngine | None = None, output_dir: str = "outputs/chats") -> None:
        self.console = Console()
        self.engine = engine or JiahaoEngine()
        self.output_dir = Path(output_dir)
        self.turns: list[dict] = []
        self.topics: list[str] = []

    # ---------- screens ----------
    def splash(self) -> None:
        self.console.clear()
        inner = max(20, self.console.width - 2 - 8)  # panel borders + horizontal padding

        width = max(len(line) for line in TITLE_ART)
        indent = " " * max(0, (inner - width) // 2)
        # Center the art as one block (per-line centering would skew rows); █ gets a gold gradient, ▒ the shadow.
        logo = Text(no_wrap=True, overflow="crop")
        for row, line in enumerate(TITLE_ART):
            logo.append(indent)
            for char in line:
                if char == "▒":
                    logo.append(char, style=SHADE_COLOR)
                else:
                    logo.append(char, style=f"bold {BANNER_COLORS[row % len(BANNER_COLORS)]}")
            logo.append("\n")
        subtitle = Align.center(Text(SUBTITLE, style=f"bold {GOLD}"))
        sign = Text.assemble(("by ", "dim"), (AUTHOR, "bold italic #FF922B"), ("   ·   ", "dim"), (VERSION, "dim"))
        quote = Text.assemble(("嘉豪：", "dim"), (f"「{random.choice(QUOTES)}」", "italic #FFD8A8"))
        body = Group(logo, subtitle, Text(""), Align.center(sign), Text(""), Align.center(quote))
        self.console.print(
            Panel(body, border_style=GOLD, padding=(1, 4), title="[bold]嘉豪语言模型[/]", subtitle="豪气冲天")
        )

    def load_model(self) -> None:
        error: list[BaseException] = []

        def worker() -> None:
            try:
                self.engine.load()
            except BaseException as exc:  # surface any load failure in the UI thread
                error.append(exc)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        with self.console.status(LOADING_LINES[0], spinner="dots12", spinner_style=GOLD) as status:
            index = 0
            while thread.is_alive():
                thread.join(timeout=2.5)
                index += 1
                status.update(f"[{GOLD}]{LOADING_LINES[index % len(LOADING_LINES)]}[/]")
        if error:
            raise error[0]
        self.console.print(f"[bold {GOLD}]✔ 嘉豪已就位。[/] [dim]输入 /help 查看玩法，/topic 让嘉豪出题。[/]\n")

    def show_help(self) -> None:
        table = Table(title="玩法", border_style=GOLD, show_header=False)
        table.add_column(style=f"bold {GOLD}")
        table.add_column()
        for command, desc in HELP_ROWS:
            table.add_row(command, desc)
        self.console.print(table)

    # ---------- chat ----------
    def type_out(self, text: str, reply) -> None:
        meters = Text.assemble(
            ("接话 ", "dim"),
            (bar(reply.relevance, 6), "#69DB7C"),
            ("   豪气值 ", "dim"),
            (bar(reply.style), GOLD),
            (f" {round(reply.style * 100)}%", f"bold {GOLD}"),
        )
        shown = Text(style="bold #FFF3BF")
        with Live(console=self.console, refresh_per_second=40, transient=False) as live:
            for char in text:
                shown.append(char)
                live.update(Panel(shown, title="[bold]嘉豪[/]", title_align="left", border_style=GOLD, padding=(0, 1)))
                time.sleep(0.025)
            live.update(
                Panel(
                    Group(shown, meters), title="[bold]嘉豪[/]", title_align="left", border_style=GOLD, padding=(0, 1)
                )
            )

    def ask(self, message: str) -> None:
        with self.console.status(f"[{GOLD}]{random.choice(THINKING_LINES)}[/]", spinner="dots", spinner_style=GOLD):
            reply = self.engine.reply(message)
        self.turns.append({"you": message, "jiahao": reply.text, "style": reply.style, "relevance": reply.relevance})
        self.type_out(reply.text, reply)
        self.show_level()

    def show_level(self) -> None:
        average = statistics.mean(t["style"] for t in self.turns)
        self.console.print(
            f"[dim]第 {len(self.turns)} 回合 · 本场豪气等级：[/][bold {GOLD}]{level_of(average)}[/]\n", justify="right"
        )

    def show_candidates(self) -> None:
        if not self.engine.last:
            self.console.print("[dim]还没有候选，先聊一句吧。[/]")
            return
        table = Table(title="本轮候选", border_style=GOLD)
        table.add_column("#", style="bold")
        table.add_column("接话", justify="right")
        table.add_column("豪气", justify="right")
        table.add_column("回复")
        for number, item in enumerate(self.engine.last.candidates, start=1):
            mark = "✓" if item["relevance"] >= 0.5 else "✗"
            table.add_row(
                str(number), f"{mark} {item['relevance']:.2f}", f"{round(item['swagger'] * 100)}%", item["text"]
            )
        self.console.print(table)
        self.console.print("[dim]输入 /pick N 改用某一句。[/]")

    def pick(self, value: str) -> None:
        reply = self.engine.pick(int(value))
        self.turns[-1].update(jiahao=reply.text, style=reply.style, relevance=reply.relevance)
        self.console.print(f"[dim]（已改用第 {value} 句）[/]")
        self.type_out(reply.text, reply)

    def like(self) -> None:
        if not self.turns:
            self.console.print("[dim]还没有可收藏的句子。[/]")
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        record = {**self.turns[-1], "liked_at": datetime.now().isoformat(timespec="seconds")}
        with (self.output_dir / "favorites.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.console.print(f"[bold {GOLD}]★ 已收藏：[/]{self.turns[-1]['jiahao']}")

    def save(self) -> Path | None:
        if not self.turns:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"嘉豪对话-{datetime.now():%Y%m%d-%H%M%S}.md"
        lines = [f"# 和嘉豪的对话 · {datetime.now():%Y-%m-%d %H:%M}", ""]
        for turn in self.turns:
            lines += [f"**你**：{turn['you']}", "", f"**嘉豪**：{turn['jiahao']}", ""]
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def suggest_topics(self) -> None:
        self.topics = random.sample(TOPICS, 3)
        for number, topic in enumerate(self.topics, start=1):
            self.console.print(f"  [bold {GOLD}]{number}[/]  {topic}")
        self.console.print("[dim]输入数字直接问嘉豪。[/]")

    def farewell(self) -> None:
        if not self.turns:
            self.console.print(f"[{GOLD}]嘉豪：你还没开口就走了？行，我允许你下次再来。[/]")
            return
        best = max(self.turns, key=lambda t: (t["style"], t["relevance"]))
        average = statistics.mean(t["style"] for t in self.turns)
        table = Table(show_header=False, border_style=GOLD, title="本场战绩")
        table.add_column(style="dim")
        table.add_column(style="bold")
        table.add_row("聊了", f"{len(self.turns)} 回合")
        table.add_row("平均豪气", f"{bar(average)} {round(average * 100)}%")
        table.add_row("豪气等级", level_of(average))
        table.add_row("最豪一句", f"「{best['jiahao']}」")
        self.console.print(table)
        saved = self.save()
        if saved:
            self.console.print(f"[dim]对话已保存到 {saved}[/]")
        self.console.print(f"[bold {GOLD}]嘉豪：今天先到这，别太想我。[/]")

    # ---------- main loop ----------
    def handle_command(self, line: str) -> bool:
        command, _, value = line.partition(" ")
        try:
            if command == "/quit":
                return False
            if command == "/help":
                self.show_help()
            elif command == "/topic":
                self.suggest_topics()
            elif command == "/all":
                self.show_candidates()
            elif command == "/pick":
                self.pick(value)
            elif command == "/like":
                self.like()
            elif command == "/temp":
                self.engine.temperature = max(0.1, min(2.0, float(value)))
                self.console.print(f"[dim]温度 = {self.engine.temperature}（越高越放飞）[/]")
            elif command == "/n":
                self.engine.candidate_count = max(1, min(16, int(value)))
                self.console.print(f"[dim]每轮候选数 = {self.engine.candidate_count}[/]")
            elif command == "/save":
                saved = self.save()
                self.console.print(f"[dim]{'已保存到 ' + str(saved) if saved else '还没有对话可保存'}[/]")
            elif command == "/reset":
                self.engine.reset()
                self.console.print("[dim]（上下文已清空，嘉豪忘记了刚才的一切）[/]")
            else:
                self.console.print("[dim]不认识这个命令，输入 /help 看看。[/]")
        except (ValueError, IndexError):
            self.console.print("[dim]用法不对，输入 /help 看看。[/]")
        return True

    def run(self) -> None:
        self.splash()
        self.load_model()
        try:
            while True:
                line = self.console.input("[bold #74C0FC]你 ›[/] ").strip()
                if not line:
                    continue
                if line.startswith("/"):
                    if not self.handle_command(line):
                        break
                    continue
                if line.isdigit() and self.topics and 1 <= int(line) <= len(self.topics):
                    line = self.topics[int(line) - 1]
                    self.console.print(f"[dim]你问：{line}[/]")
                self.topics = []
                self.ask(line)
        except (KeyboardInterrupt, EOFError):
            self.console.print()
        self.farewell()


def main() -> None:
    prepare_console()
    import argparse

    from src.app.engine import DEFAULT_ADAPTER, DEFAULT_MODEL_CONFIG

    parser = argparse.ArgumentParser(description="Jiahao Language Model chat")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    JiahaoApp(JiahaoEngine(args.adapter, args.model_config, args.temperature)).run()


if __name__ == "__main__":
    main()
