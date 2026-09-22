# Ollama benchmark

Run **Benchmark Ollama** in VS Code's Run and Debug panel, or run `python benchmark.py` from the repo root. Enter the Ollama server URL (default `http://localhost:11434`), choose models and prompts by number or `all`, then enter a positive round count (default 5). The script creates any missing standard prompts after it connects to Ollama. Run `python generate_prompts.py --overwrite` to regenerate all standard prompts.

The layout is `prompt/*.txt` for inputs and `fqdn/<endpoint-host>/<YYYY-MM-DD>/` for results. Each run gets a timestamped Markdown report, run metadata JSON, a copy of each selected prompt, and one response text file plus metadata JSON per completed sample. The report updates after each sample so interrupted runs retain their completed results.

Standard prompt filenames specify their exact UTF-8 byte lengths: `100.txt`, `1000.txt`, and `10000.txt`. Each prompt contains a narrative fact and a question about it.

TTFT is measured by the client from request start to the first streamed response text, including model load. Prefill and evaluation tokens per second use Ollama's reported counts and nanosecond durations. MTP is reported as unknown unless model information or response fields provide evidence.

The runner calls `/api/show` with `verbose: true` to inspect model metadata and MTP tensors. An `on` label means drafting is configured for a model with an MTP head; the draft counters are needed to confirm activity in an individual request.

Ollama 0.34.2 omits speculative draft counts from `/api/generate`, even though its underlying `llama-server` computes them. The benchmark checks the API response first, then an existing local Ollama log when using a local endpoint. For a remote endpoint, set `OLLAMA_SERVER_LOG` before launching if its log is accessible. Supported values are a local or shared file path, `ssh://host/absolute/path`, `journal://host/ollama` for a systemd unit, and `docker://host/container`. SSH sources use the installed `ssh` command and an existing noninteractive login. The log must contain the `draft acceptance = ... (accepted / generated)` line from llama-server or `speculative decode stats` from Ollama's MLX runner. The script reads only entries written during each request and records drafted and accepted counts, their source, and the computed acceptance rate in the sample JSON. If several acceptance entries arrive in the same request window, the rate remains `n/a` to avoid assigning another request's counters. Benchmark setup adds no log-related prompt.
