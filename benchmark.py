"""Interactive Ollama benchmark. Run from VS Code or with `python benchmark.py`."""

from __future__ import annotations

import json
import hashlib
import os
import re
import socket
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from acceptance import LogReader, parse_log_counters
from generate_prompts import ROOT, SIZES, generate


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def endpoint_url(value: str) -> tuple[str, str]:
    value = value.strip() or "http://localhost:11434"
    if "://" not in value:
        value = "http://" + value
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Enter an http(s) Ollama host or URL.")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ValueError("Enter the Ollama server root URL, without a path or credentials.")
    host = parsed.hostname.lower().rstrip(".")
    fqdn = socket.getfqdn(host).lower().rstrip(".") if host in ("localhost", "127.0.0.1", "::1") else host
    if not fqdn or not re.fullmatch(r"[a-z0-9._-]+", fqdn):
        raise ValueError("The endpoint host cannot be used as an output folder name.")
    return value.rstrip("/"), fqdn


def local_log_source(endpoint: str) -> str | None:
    """Use an existing local Ollama log when the API host is this machine."""
    host = urlsplit(endpoint).hostname
    if host not in ("localhost", "127.0.0.1", "::1", socket.gethostname().lower()):
        return None
    candidates = [Path.home() / ".ollama" / "logs" / "server.log"]
    if os.environ.get("LOCALAPPDATA"):
        candidates.append(Path(os.environ["LOCALAPPDATA"]) / "Ollama" / "server.log")
    candidates.append(Path("/var/log/ollama.log"))
    return next((str(path) for path in candidates if path.is_file()), None)


def request_json(url: str, payload: dict | None = None, timeout: int = 30) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object from {url}")
    return result


def choose(items: list[str], label: str) -> list[str]:
    print(f"\nAvailable {label}:")
    for index, item in enumerate(items, 1):
        print(f"  {index}. {item}")
    while True:
        answer = input(f"Choose {label} number(s), comma separated, or 'all' [all]: ").strip().lower()
        if answer in ("", "all"):
            return items
        try:
            indices = [int(part.strip()) for part in answer.split(",")]
            if indices and all(1 <= index <= len(items) for index in indices):
                return [items[index - 1] for index in dict.fromkeys(indices)]
        except ValueError:
            pass
        print("Enter listed numbers or all.")


def rounds_input() -> int:
    while True:
        answer = input("Rounds per model and prompt [5]: ").strip()
        try:
            rounds = int(answer) if answer else 5
            if rounds > 0:
                return rounds
        except ValueError:
            pass
        print("Enter a positive whole number.")


def safe_filename(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "item"
    return stem[:80] + "_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


def number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def rate(count: object, duration_ns: object) -> float | None:
    c, d = number(count), number(duration_ns)
    return c * 1e9 / d if c is not None and d is not None and d > 0 else None


def draft_counters(final: dict) -> tuple[int, int] | None:
    """Accept both native top-level counters and llama.cpp's timings shape."""
    for fields in (final.get("timings"), final):
        if not isinstance(fields, dict):
            continue
        drafted, accepted = number(fields.get("draft_n")), number(fields.get("draft_n_accepted"))
        if drafted is not None and accepted is not None and drafted > 0 and 0 <= accepted <= drafted:
            return int(drafted), int(accepted)
    return None


def mtp_metrics(final: dict, model_info: dict, model: str, log_counts: tuple[int, int] | None = None) -> tuple[str, float | None]:
    """Only mark MTP known when model configuration and response support it."""
    params = str(model_info.get("parameters", ""))
    match = re.search(r"(?m)^\s*draft_num_predict\s+(\d+)\s*$", params)
    model_text = model.lower() + " " + json.dumps(model_info.get("details", {})).lower() + " " + str(model_info.get("modelfile", "")).lower()
    gguf_info = model_info.get("model_info", {})
    head_in_metadata = isinstance(gguf_info, dict) and any(
        key.endswith(".nextn_predict_layers") and number(value) is not None and number(value) > 0
        for key, value in gguf_info.items()
    )
    head_in_tensors = any(
        ".nextn." in tensor.get("name", "").lower() or ".mtp." in tensor.get("name", "").lower()
        for tensor in model_info.get("tensors", []) if isinstance(tensor, dict)
    )
    hints_mtp = head_in_metadata or head_in_tensors or "mtp" in model_text
    state = "on" if match and int(match.group(1)) > 0 and hints_mtp else "off" if match and int(match.group(1)) == 0 else "unknown"
    counters = draft_counters(final) or log_counts
    acceptance = counters[1] / counters[0] if counters and hints_mtp else None
    if acceptance is not None:
        state = "on"
    return state, acceptance


def run_one(endpoint: str, model: str, prompt: str, model_info: dict, log_reader: LogReader | None = None) -> tuple[str, dict]:
    payload = {"model": model, "prompt": prompt, "stream": True}
    request = Request(endpoint + "/api/generate", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    log_mark = None
    log_error = None
    if log_reader is not None and ("mtp" in model.lower() or mtp_metrics({}, model_info, model)[0] == "on"):
        try:
            log_mark = log_reader.mark()
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            log_error = f"Cannot mark server log: {exc}"
    started_at = now()
    start = time.perf_counter()
    fragments: list[str] = []
    first_token_s = None
    final = None
    with urlopen(request, timeout=3600) as response:
        for line in response:
            if not line.strip():
                continue
            event = json.loads(line)
            if "error" in event:
                raise RuntimeError(str(event["error"]))
            fragment = event.get("response", "")
            if fragment:
                fragments.append(fragment)
                if first_token_s is None:
                    first_token_s = time.perf_counter() - start
            if event.get("done"):
                final = event
                break
    wall_s = time.perf_counter() - start
    if final is None:
        raise RuntimeError("Ollama stream ended without a final metrics event")
    api_counts = draft_counters(final)
    log_counts = None
    if log_mark is not None and api_counts is None:
        for attempt in range(5):
            try:
                log_counts = parse_log_counters(log_reader.read_since(log_mark))
                if log_counts is not None:
                    break
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                log_error = f"Cannot read server log: {exc}"
                break
            if attempt < 4:
                time.sleep(0.25)
    counters = api_counts or log_counts
    mtp, acceptance = mtp_metrics(final, model_info, model, log_counts)
    metrics = {
        "started_at": started_at,
        "finished_at": now(),
        "wall_s": wall_s,
        "ttft_s": first_token_s,
        "prefill_tps": rate(final.get("prompt_eval_count"), final.get("prompt_eval_duration")),
        "eval_tps": rate(final.get("eval_count"), final.get("eval_duration")),
        "mtp": mtp,
        "mtp_acceptance_rate": acceptance,
        "drafted_tokens": counters[0] if counters else None,
        "accepted_draft_tokens": counters[1] if counters else None,
        "acceptance_source": "api" if api_counts else log_reader.source if log_counts else None,
        "acceptance_error": log_error,
    }
    return "".join(fragments), {"request": payload, "ollama_final": final, "metrics": metrics}


def summarize(values: list[float | None]) -> str:
    present = [v for v in values if v is not None]
    if not present:
        return "n/a"
    return f"{statistics.mean(present):.2f} / {statistics.median(present):.2f} / {min(present):.2f}–{max(present):.2f} (n={len(present)})"


def report(run: dict, samples: list[dict]) -> str:
    lines = [
        "# Ollama benchmark",
        "",
        f"- FQDN: `{run['fqdn']}`",
        f"- Endpoint: `{run['endpoint']}`",
        f"- Started: {run['started_at']}",
        f"- Finished: {run['finished_at']}",
        f"- Rounds per combination: {run['rounds']}",
        "",
        "TTFT is local time from request start to the first nonempty streamed response. It includes model loading and prompt processing. "
        "Prefill and evaluation rates use Ollama token counts and durations (nanoseconds). "
        "Prefill may include cached tokens in its count; see raw metadata. "
        "MTP `on` means the model configuration enables drafting and indicates an MTP head; it does not prove that a particular request drafted tokens. "
        "Acceptance uses draft counts from the API when available, or the configured Ollama server log. "
        "A missing rate means no unambiguous counters were available for that sample. "
        "The per-sample JSON records the counter source and any log error.",
        "",
    ]
    for model in run["models"]:
        lines += [f"## {model}", ""]
        lines += ["| Prompt | Runs | TTFT s | Prefill tok/s | Eval tok/s | MTP | Acceptance % |", "| --- | ---: | ---: | ---: | ---: | --- | ---: |"]
        for prompt in run["prompts"]:
            group = [s for s in samples if s["model"] == model and s["prompt"] == prompt and "metrics" in s]
            if not group:
                lines.append(f"| {prompt} | 0 | n/a | n/a | n/a | n/a | n/a |")
                continue
            metrics = [s["metrics"] for s in group]
            mtp_states = sorted({m["mtp"] for m in metrics})
            fmt = lambda key: summarize([m[key] for m in metrics])
            acceptance = summarize([m["mtp_acceptance_rate"] * 100 if m["mtp_acceptance_rate"] is not None else None for m in metrics])
            lines.append(f"| {prompt} | {len(group)} | {fmt('ttft_s')} | {fmt('prefill_tps')} | {fmt('eval_tps')} | {', '.join(mtp_states)} | {acceptance} |")
        lines += [""]
    lines += ["Values in numeric cells are mean / median / min–max with sample count. Raw response and metadata files are alongside this report.", ""]
    failures = [s for s in samples if "error" in s]
    if failures:
        lines += ["## Failures", ""]
        for s in failures:
            lines.append(f"- {s['model']} / {s['prompt']} / round {s['round']}: {s['error']}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    while True:
        try:
            endpoint, fqdn = endpoint_url(input("Ollama endpoint [http://localhost:11434]: "))
            tags = request_json(endpoint + "/api/tags")
            models = sorted({m["name"] for m in tags.get("models", []) if isinstance(m.get("name"), str)})
            if not models:
                raise ValueError("No models returned by /api/tags.")
            break
        except (ValueError, HTTPError, URLError, TimeoutError, OSError) as exc:
            print(f"Cannot use endpoint: {exc}")
    missing = [size for size in SIZES if not (ROOT / "prompt" / f"{size}.txt").exists()]
    if missing:
        print(f"Missing prompts: {', '.join(f'{size}.txt' for size in missing)}. Generating them now.")
        generate()
    selected_models = choose(models, "models")
    prompts = sorted((p for p in (ROOT / "prompt").glob("*.txt") if p.is_file() and p.name != "0.txt"), key=lambda p: (not p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else p.name))
    selected_prompts = choose([p.name for p in prompts], "prompts")
    rounds = rounds_input()
    start = datetime.now().astimezone()
    run_id = start.strftime("%H%M%S") + f"-{time.time_ns() % 1_000_000_000:09d}"
    output = ROOT / "fqdn" / fqdn / start.strftime("%Y-%m-%d")
    output.mkdir(parents=True, exist_ok=True)
    prompt_snapshots = {}
    for prompt_name in selected_prompts:
        source = ROOT / "prompt" / prompt_name
        content = source.read_bytes()
        snapshot = f"{run_id}_{safe_filename(prompt_name)}_prompt.txt"
        (output / snapshot).write_bytes(content)
        prompt_snapshots[prompt_name] = {"snapshot_file": snapshot, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    model_info = {}
    for model in selected_models:
        try:
            model_info[model] = request_json(endpoint + "/api/show", {"model": model, "verbose": True})
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            model_info[model] = {"show_error": str(exc)}
    source = os.environ.get("OLLAMA_SERVER_LOG", "").strip() or local_log_source(endpoint)
    log_reader = LogReader(source) if source else None
    run = {"fqdn": fqdn, "endpoint": endpoint, "started_at": now(), "models": selected_models,
           "prompts": selected_prompts, "prompt_snapshots": prompt_snapshots,
           "rounds": rounds, "server_log_source": source or None, "model_list": tags, "model_info": model_info}
    samples: list[dict] = []
    total = len(selected_models) * len(selected_prompts) * rounds
    for model in selected_models:
        for prompt_name in selected_prompts:
            prompt_text = (ROOT / "prompt" / prompt_name).read_text(encoding="utf-8")
            for round_index in range(1, rounds + 1):
                stem = f"{run_id}_{safe_filename(model)}_{safe_filename(prompt_name)}_r{round_index:03d}"
                print(f"[{len(samples) + 1}/{total}] {model} / {prompt_name} / round {round_index}", flush=True)
                item = {"model": model, "prompt": prompt_name, "round": round_index, "response_file": stem + ".txt"}
                try:
                    response, metadata = run_one(endpoint, model, prompt_text, model_info[model], log_reader)
                    (output / item["response_file"]).write_text(response, encoding="utf-8")
                    item.update(metadata)
                    item["metrics"] = metadata["metrics"]
                    if metadata["metrics"]["acceptance_error"]:
                        print(f"  Acceptance unavailable: {metadata['metrics']['acceptance_error']}")
                except (HTTPError, URLError, TimeoutError, OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
                    item["error"] = str(exc)
                    print(f"  Failed: {exc}")
                (output / (stem + ".json")).write_text(json.dumps(item, indent=2, ensure_ascii=False), encoding="utf-8")
                samples.append(item)
                run["finished_at"] = now()
                (output / f"{run_id}_run.json").write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
                (output / f"{run_id}_report.md").write_text(report(run, samples), encoding="utf-8")
    print(f"\nReport: {output / (run_id + '_report.md')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Completed samples remain saved.")
