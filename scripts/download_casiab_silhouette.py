from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "http://www.cbsr.ia.ac.cn/GaitDatasetB-silh.zip"


@dataclass(slots=True)
class DownloadReport:
    url: str
    output_path: str
    existed_before: bool
    bytes_before: int
    bytes_after: int
    remote_content_length: int | None
    file_exists: bool
    size_reasonable: bool
    status: str
    attempts: int
    downloader: str
    command: str
    stderr_tail: str
    started_at: str
    finished_at: str
    duration_seconds: float


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="下载 CASIA-B silhouette 官方压缩包，支持断点续传与最小重试")
    parser.add_argument("--url", default=DEFAULT_URL, help="下载链接")
    parser.add_argument(
        "--output-path",
        default=str(ROOT / "downloads" / "GaitDatasetB-silh.zip"),
        help="压缩包输出路径",
    )
    parser.add_argument(
        "--reports-dir",
        default=str(ROOT / "reports"),
        help="报告输出目录",
    )
    parser.add_argument("--json-report", default="casia_b_download.json", help="JSON 报告文件名")
    parser.add_argument("--md-report", default="casia_b_download.md", help="Markdown 报告文件名")
    parser.add_argument("--timeout", type=int, default=30, help="探测请求超时秒数")
    parser.add_argument("--retries", type=int, default=2, help="失败后的最小重试次数")
    parser.add_argument(
        "--min-reasonable-bytes",
        type=int,
        default=100 * 1024 * 1024,
        help="判断下载结果是否至少达到“看起来合理”的最小字节数，默认 100 MiB",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_path = Path(args.output_path).expanduser().resolve()
    reports_dir = Path(args.reports_dir).expanduser().resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    remote_size = probe_remote_content_length(args.url, timeout=args.timeout)
    report = download_with_resume(
        url=args.url,
        output_path=output_path,
        remote_content_length=remote_size,
        retries=args.retries,
        min_reasonable_bytes=args.min_reasonable_bytes,
    )

    json_path = reports_dir / args.json_report
    md_path = reports_dir / args.md_report
    json_path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print(f"status: {report.status}")
    print(f"output_path: {report.output_path}")
    print(f"bytes_after: {report.bytes_after}")
    print(f"remote_content_length: {report.remote_content_length}")
    print(f"attempts: {report.attempts}")
    print(f"json_report: {json_path}")
    print(f"md_report: {md_path}")

    if report.status != "downloaded":
        raise SystemExit(1)


def probe_remote_content_length(url: str, *, timeout: int) -> int | None:
    curl = shutil.which("curl")
    if curl is None:
        return None
    completed = subprocess.run(
        [curl, "-L", "--head", "--silent", "--show-error", "--max-time", str(timeout), url],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            raw = line.split(":", 1)[1].strip()
            try:
                return int(raw)
            except ValueError:
                return None
    return None


def download_with_resume(
    *,
    url: str,
    output_path: Path,
    remote_content_length: int | None,
    retries: int,
    min_reasonable_bytes: int,
) -> DownloadReport:
    downloader = choose_downloader()
    if downloader is None:
        raise SystemExit("当前系统找不到 curl 或 wget，无法下载 CASIA-B silhouette")

    existed_before = output_path.exists()
    bytes_before = output_path.stat().st_size if output_path.exists() else 0
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    start_time = time.time()

    stderr_tail = ""
    last_command = ""
    attempts = 0
    success = False

    for attempt in range(1, retries + 2):
        attempts = attempt
        command = build_download_command(downloader, url, output_path)
        last_command = shlex.join(command)
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        stderr_tail = tail_text("\n".join(part for part in [completed.stdout, completed.stderr] if part), max_chars=4000)
        file_size = output_path.stat().st_size if output_path.exists() else 0
        complete_by_length = remote_content_length is not None and file_size == remote_content_length
        if completed.returncode == 0 and output_path.exists() and (complete_by_length or file_size >= min_reasonable_bytes):
            success = True
            break
        time.sleep(min(5, attempt))

    bytes_after = output_path.stat().st_size if output_path.exists() else 0
    finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
    size_reasonable = bytes_after >= min_reasonable_bytes and (
        remote_content_length is None or bytes_after == remote_content_length
    )
    status = "downloaded" if success and size_reasonable else "blocked"
    return DownloadReport(
        url=url,
        output_path=str(output_path),
        existed_before=existed_before,
        bytes_before=bytes_before,
        bytes_after=bytes_after,
        remote_content_length=remote_content_length,
        file_exists=output_path.exists(),
        size_reasonable=size_reasonable,
        status=status,
        attempts=attempts,
        downloader=downloader,
        command=last_command,
        stderr_tail=stderr_tail,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=round(time.time() - start_time, 2),
    )


def choose_downloader() -> str | None:
    if shutil.which("curl"):
        return "curl"
    if shutil.which("wget"):
        return "wget"
    return None


def build_download_command(downloader: str, url: str, output_path: Path) -> list[str]:
    if downloader == "curl":
        return [
            "curl",
            "-L",
            "-C",
            "-",
            "--fail",
            "--retry",
            "2",
            "--retry-delay",
            "2",
            "--output",
            str(output_path),
            url,
        ]
    return [
        "wget",
        "-c",
        "--tries=3",
        "--waitretry=2",
        "-O",
        str(output_path),
        url,
    ]


def tail_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def render_markdown(report: DownloadReport) -> str:
    lines = [
        "# CASIA-B 下载报告",
        "",
        f"- URL：`{report.url}`",
        f"- 输出文件：`{report.output_path}`",
        f"- 下载器：`{report.downloader}`",
        f"- 状态：`{report.status}`",
        f"- 启动时间：`{report.started_at}`",
        f"- 完成时间：`{report.finished_at}`",
        f"- 耗时：`{report.duration_seconds}` 秒",
        f"- 下载前是否存在：`{report.existed_before}`",
        f"- 下载前大小：`{report.bytes_before}`",
        f"- 下载后大小：`{report.bytes_after}`",
        f"- 远端 Content-Length：`{report.remote_content_length}`",
        f"- 大小是否看起来合理：`{report.size_reasonable}`",
        f"- 尝试次数：`{report.attempts}`",
        "",
        "## 实际命令",
        "",
        "```bash",
        report.command,
        "```",
        "",
        "## 输出尾部",
        "",
        "```text",
        report.stderr_tail or "(empty)",
        "```",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
