# Ollama benchmark

Run **Benchmark Ollama** in VS Code's Run and Debug panel, or run `python benchmark.py` from the repo root. Enter the Ollama server URL (default `http://localhost:11434`), choose models and prompts by number or `all`, then enter a positive round count (default 5). The script creates any missing standard prompts after it connects to Ollama. Run `python generate_prompts.py --overwrite` to regenerate all standard prompts.

The layout is `prompt/*.txt` for inputs and `fqdn/<endpoint-host>/<YYYY-MM-DD>/` for results. Each run gets a timestamped Markdown report, run metadata JSON, a copy of each selected prompt, and one response text file plus metadata JSON per completed sample. The report updates after each sample so interrupted runs retain their completed results.

Standard prompt filenames specify their exact UTF-8 byte lengths: `0.txt`, `100.txt`, `1000.txt`, and `10000.txt`. `0.txt` is an empty baseline; it cannot contain instructions, a narrative, and a question. The other prompts contain a narrative fact and a question about it.

TTFT is measured by the client from request start to the first streamed response text, including model load. Prefill and evaluation tokens per second use Ollama's reported counts and nanosecond durations. MTP is reported as unknown unless model information or response fields provide evidence. The API may not expose MTP acceptance counts; in that case the report shows `n/a`.
