#!/usr/bin/env python3
"""Control the travel-itinerary collector on the configured SSH server."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shlex
from pathlib import Path

import paramiko


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_COLLECTOR = PROJECT_ROOT / "scripts" / "collect_travel_itineraries.py"
REMOTE_DIR = "/root/autodl-tmp/travel-rag-collection"
REMOTE_SCRIPT = f"{REMOTE_DIR}/collect_travel_itineraries.py"
REMOTE_OUTPUT = f"{REMOTE_DIR}/travel_itineraries_50.jsonl"
REMOTE_REPORT = f"{REMOTE_DIR}/travel_itineraries_50.report.json"
REMOTE_LOG = f"{REMOTE_DIR}/collector.log"
REMOTE_PID = f"{REMOTE_DIR}/collector.pid"
REMOTE_PYTHON_CANDIDATES = [
    "/root/miniconda3/bin/python",
    "/root/miniconda3/bin/python3",
    "/root/anaconda3/bin/python",
    "/usr/bin/python3",
]


def connect(args: argparse.Namespace) -> paramiko.SSHClient:
    password = os.environ.get("SEETACLOUD_SSH_PASSWORD") or getpass.getpass(
        f"SSH password for {args.ssh_user}@{args.ssh_host}: "
    )
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.ssh_host,
        port=args.ssh_port,
        username=args.ssh_user,
        password=password,
        allow_agent=False,
        look_for_keys=False,
        timeout=20,
    )
    return client


def run(client: paramiko.SSHClient, command: str, timeout: int = 60) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    code = stdout.channel.recv_exit_status()
    return code, stdout.read().decode("utf-8", "replace"), stderr.read().decode("utf-8", "replace")


def find_remote_python(client: paramiko.SSHClient) -> str | None:
    tests = " ".join(shlex.quote(path) for path in REMOTE_PYTHON_CANDIDATES)
    command = f"for p in {tests}; do if [ -x \"$p\" ]; then echo \"$p\"; exit 0; fi; done; exit 1"
    code, stdout, _ = run(client, f"bash -lc {shlex.quote(command)}")
    return stdout.strip() if code == 0 else None


def action_check(client: paramiko.SSHClient) -> int:
    python = find_remote_python(client)
    if not python:
        print(json.dumps({"ready": False, "error": "No supported Python executable found"}))
        return 1
    command = f"{shlex.quote(python)} -c 'import requests, bs4; print(requests.__version__); print(bs4.__version__)'"
    code, stdout, stderr = run(client, command)
    print(json.dumps({"ready": code == 0, "python": python, "stdout": stdout, "stderr": stderr}, ensure_ascii=False))
    return code


def action_prepare(client: paramiko.SSHClient) -> int:
    python = find_remote_python(client)
    if not python:
        print("No supported Python executable found")
        return 1
    command = f"{shlex.quote(python)} -m pip install --disable-pip-version-check requests beautifulsoup4"
    code, stdout, stderr = run(client, command, timeout=300)
    print(stdout, end="")
    print(stderr, end="")
    return code


def action_probe(client: paramiko.SSHClient) -> int:
    python = find_remote_python(client)
    if not python:
        print("No supported Python executable found")
        return 1
    snippet = (
        "import json,time,requests; out=[]; "
        "urls=['https://www.google.com/search?q=travel+itinerary&num=10',"
        "'https://www.baidu.com/s?wd=%E6%97%85%E6%B8%B8%E8%B7%AF%E4%B9%A6']; "
        "\nfor u in urls:\n"
        " t=time.time()\n"
        " try:\n"
        "  r=requests.get(u,timeout=8,headers={'User-Agent':'Mozilla/5.0'}); "
        "out.append({'url':u,'status':r.status_code,'bytes':len(r.content),'seconds':round(time.time()-t,2),'final_url':r.url})\n"
        " except Exception as e: out.append({'url':u,'error':repr(e),'seconds':round(time.time()-t,2)})\n"
        "print(json.dumps(out,ensure_ascii=False))"
    )
    command = f"{shlex.quote(python)} -c {shlex.quote(snippet)}"
    code, stdout, stderr = run(client, command, timeout=30)
    print(stdout, end="")
    print(stderr, end="")
    return code


def action_start(client: paramiko.SSHClient, args: argparse.Namespace) -> int:
    python = find_remote_python(client)
    if not python:
        print("No supported Python executable found")
        return 1
    code, _, stderr = run(client, f"mkdir -p {shlex.quote(REMOTE_DIR)}")
    if code:
        print(stderr)
        return code
    with client.open_sftp() as sftp:
        sftp.put(str(LOCAL_COLLECTOR), REMOTE_SCRIPT)
    command = " ".join(
        [
            "nohup", shlex.quote(python), "-u", shlex.quote(REMOTE_SCRIPT),
            "--output", shlex.quote(REMOTE_OUTPUT),
            "--report", shlex.quote(REMOTE_REPORT),
            "--google-target", str(args.google_target),
            "--baidu-target", str(args.baidu_target),
            "--pages-per-query", str(args.pages_per_query),
            "--delay", str(args.delay),
            ">", shlex.quote(REMOTE_LOG), "2>&1", "&", "echo", "$!", ">", shlex.quote(REMOTE_PID),
        ]
    )
    code, stdout, stderr = run(client, f"bash -lc {shlex.quote(command)}")
    print(json.dumps({"started": code == 0, "stdout": stdout, "stderr": stderr}, ensure_ascii=False))
    return code


def action_status(client: paramiko.SSHClient) -> int:
    command = (
        f"pid=$(cat {shlex.quote(REMOTE_PID)} 2>/dev/null || true); "
        "running=false; if [ -n \"$pid\" ] && kill -0 \"$pid\" 2>/dev/null; then running=true; fi; "
        "printf 'RUNNING=%s\\nPID=%s\\n' \"$running\" \"$pid\"; "
        "if [ -n \"$pid\" ]; then ps -p \"$pid\" -o pid=,etime=,stat=,cmd= 2>/dev/null || true; fi; "
        f"wc -l -c {shlex.quote(REMOTE_OUTPUT)} 2>/dev/null || true; "
        f"test -f {shlex.quote(REMOTE_REPORT)} && cat {shlex.quote(REMOTE_REPORT)} || true; "
        "printf '\\n---LOG---\\n'; "
        f"tail -n 30 {shlex.quote(REMOTE_LOG)} 2>/dev/null || true"
    )
    code, stdout, stderr = run(client, f"bash -lc {shlex.quote(command)}")
    print(stdout, end="")
    print(stderr, end="")
    return code


def action_stop(client: paramiko.SSHClient) -> int:
    command = (
        f"pid=$(cat {shlex.quote(REMOTE_PID)} 2>/dev/null || true); "
        "if [ -z \"$pid\" ]; then echo 'No collector PID found'; exit 0; fi; "
        "if kill -0 \"$pid\" 2>/dev/null; then kill -TERM \"$pid\"; echo \"Stopped collector PID $pid\"; "
        "else echo \"Collector PID $pid is not running\"; fi"
    )
    code, stdout, stderr = run(client, f"bash -lc {shlex.quote(command)}")
    print(stdout, end="")
    print(stderr, end="")
    return code


def action_download(client: paramiko.SSHClient, local_output: Path) -> int:
    local_output.parent.mkdir(parents=True, exist_ok=True)
    local_report = local_output.with_suffix(".report.json")
    with client.open_sftp() as sftp:
        sftp.get(REMOTE_OUTPUT, str(local_output))
        sftp.get(REMOTE_REPORT, str(local_report))
    print(json.dumps({"output": str(local_output), "report": str(local_report)}, ensure_ascii=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "prepare", "probe", "start", "status", "stop", "download"])
    parser.add_argument("--ssh-host", default="connect.cqa1.seetacloud.com")
    parser.add_argument("--ssh-port", type=int, default=38174)
    parser.add_argument("--ssh-user", default="root")
    parser.add_argument("--google-target", type=int, default=25)
    parser.add_argument("--baidu-target", type=int, default=25)
    parser.add_argument("--pages-per-query", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.7)
    parser.add_argument(
        "--local-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "travel_itineraries_50.jsonl",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = connect(args)
    try:
        if args.action == "check":
            return action_check(client)
        if args.action == "prepare":
            return action_prepare(client)
        if args.action == "probe":
            return action_probe(client)
        if args.action == "start":
            return action_start(client, args)
        if args.action == "status":
            return action_status(client)
        if args.action == "stop":
            return action_stop(client)
        return action_download(client, args.local_output)
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
