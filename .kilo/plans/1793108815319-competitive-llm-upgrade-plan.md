# CoreAI3D Competitive LLM Upgrade Plan

## Goal
Replace hardcoded rule-based responses in `CoreAI3D/src/Language.cpp` with a hybrid architecture: fast C++ paths for greetings/math/lists plus a local LLM (Ollama sidecar) with RAG-style web search context injection.

## Decisions
- Provider: Ollama sidecar over HTTP (libcurl already available).
- Response mode: synchronous first iteration; keep Next.js JSON fallback.
- Default model: `qwen2.5:7b-instruct`.
- Fast-path rules remain for greetings, code examples, and `generateList`.
- Web snippets are injected as RAG context before generation.

## Affected Files
- `CoreAI3D/src/Language.cpp` — `respond()`, `shouldSearchWeb()`, `webSearch()`, `summarizeWebResult()`
- `CoreAI3D/src/APIServer.cpp` / `include/APIServer.hpp` — LLM config
- `CoreAI3D/src/Core.cpp` — unchanged
- `web/app/api/chat/route.ts` — no major changes

## Config
Add to `APIServer.hpp` and load from `config.json` in `APIServer.cpp`:
- `llmProvider` (default `ollama`)
- `llmBaseUrl` (default `http://localhost:11434`)
- `llmModel` (default `qwen2.5:7b-instruct`)
- `llmTemperature` (default `0.7`)
- `llmMaxTokens` (default `512`)

## Implementation Steps
1. Add config members and parsing.
2. Add `Language::callOllama(const std::string& prompt, const std::string& context)` using libcurl POST to `/api/generate` with `stream:false`.
3. In `respond()`:
   - Keep existing fast paths first.
   - If `shouldSearchWeb(message)` is true, call `webSearch()` and build RAG prompt.
   - Otherwise build direct prompt.
   - Call Ollama; on failure/timeout, fall back to concise direct response or web summary.
4. Return response string; update `getLastResponseUsedWebSearch()` and `getLastWebSearchQuery()`.
5. Add Docker Compose Ollama service and model pull instructions.
6. Keep `Core.cpp` unchanged.

## RAG Prompt Template
```
System: You are CoreAI3D. Answer directly and helpfully.
Context: <web snippets or none>
User: <message>
```

## Failure Modes
- Ollama unreachable: no crash; return fallback response.
- Empty model output: return "I couldn't generate a response right now."
- Web search empty: still answer from model without context.
- Config missing: use defaults.

## Rollout
- Document Ollama setup in README/deploy notes.
- Pull model before first run.
- Rebuild backend and frontend, restart services.

## Validation
- `g++ -fsyntax-only -std=c++17 -I CoreAI3D/include CoreAI3D/src/Language.cpp CoreAI3D/src/APIServer.cpp`
- `cd web && npx tsc --noEmit`
- Manual chat: list queries, greeting, open-ended factual query, Ollama-down fallback.

## Open Questions
- None.
