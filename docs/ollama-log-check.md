# Ollama server-log check (M3 Step 0)

Done by Claude Code on 2026-11-11 (prompt M3-01, Aeverum's open item, D-043).

**Result: at Ollama's normal log level, no memory text reaches the server log.** One made-up pair was sent to `m1-judge:v1`. Its marker word, the person's name and the pet word appear 0 times in the log. The request added only one line about the request itself: the time, the status code 200, how long it took, the address `127.0.0.1` and the path `/api/chat`.

## Where the log is on this Mac

- **Ollama's docs** (https://docs.ollama.com/troubleshooting, read 2026-11-11) say: on a Mac the log is `~/.ollama/logs/server.log`, and when you start `ollama serve` yourself in a terminal, the log goes to that terminal. The same page and https://docs.ollama.com/faq say nothing about whether prompts are logged, and name no setting to switch prompt logging off.
- **This Mac is set up differently.** Ollama 0.24.0 was installed with Homebrew and runs as a Homebrew service, not as the Mac app. `~/.ollama/logs/` does not exist. The service's settings file, `~/Library/LaunchAgents/homebrew.mxcl.ollama.plist`, sends everything the server prints to **`/opt/homebrew/var/log/ollama.log`**. That service (process 23439) is the one that answers on `127.0.0.1:11434` (checked with `lsof`). It has two extra settings (`OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`) and no `OLLAMA_DEBUG`.
- A second `ollama serve` (process 22707, `/opt/homebrew/bin/ollama`) was also running. It does not answer on port 11434, and what it prints goes to `/dev/null` (nowhere), so it keeps no log.

## What was done

1. Log size before: 5,048,456 bytes. The marker word appeared 0 times.
2. One made-up pair was sent the way `docs/run-judge-on-mac.md` says: `/api/chat`, `"stream": false`, the exact system message from `src/judge/prompt.py` (prompt SHA-256 `83d14d39...2865c`), no `options`. The person was "Tovik Ashmerrin", and both memories held the marker word "Quorbelvane" (a ferret's name in one, a boat's name in the other). All of it was made up.
3. The judge answered `contradiction` (a normal answer).
4. Log size after: 5,053,468 bytes (5,012 bytes more). Searching the whole log: "Quorbelvane" 0, "Tovik" 0, "Ashmerrin" 0, "ferret" 0. The new lines are about loading the model (file, memory, GPU layers) plus one request line:
   `[GIN] 2026/11/11 - 14:15:09 | 200 |  3.865991791s |       127.0.0.1 | POST     "/api/chat"`

## Is there a setting that would add memory text?

Nothing in the log needs switching off, so no setting was tried. One open point: Ollama's docs mention a debug setting, `OLLAMA_DEBUG=1`, for "additional debug logging", without saying what it adds. Whether debug mode writes prompt text was **not tested**. To be safe, keep `OLLAMA_DEBUG` unset on any machine that judges real memories. It is unset here today.

## What this means for Aeverum

With Ollama set up as on this Mac, judging a pair leaves no memory text in Ollama's server log. On another machine, check where its log goes (the Mac app writes to `~/.ollama/logs/server.log`) and that `OLLAMA_DEBUG` is unset.
