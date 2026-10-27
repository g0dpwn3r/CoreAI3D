# Chat Quality + Local LLM Capability Plan

## Part 1: Chat Quality Regression — IMPLEMENTED

All fixes are complete and compile cleanly with `g++ -fsyntax-only -std=c++17`. No source changes remain.

**`CoreAI3D/src/Language.cpp`**
- Removed all refusal/coaching language ("Could you rephrase?", "What specific factors matter most to you?", "I don't have a pre-built list", etc.)
- Added `generateList` lambda: hardcoded lists for 9 categories (animals, girls names, programming languages, countries, fruits, colors, cars, planets) with starting-letter filtering (`letter X` and `start with X` patterns)
- Expanded `shouldSearchWeb()` triggers to include `list`, `name`, `examples of`, `give me`, `show me`, `tell me`, `how many`, `facts about`, etc.
- Added `#include <sstream>` for `std::ostringstream`

**`web/components/ChatWindow.tsx`**
- Removed `hasSwitchedToGenerating` flag and all `setStreamState('generating')` calls
- `streamState` now goes `'thinking'` → `'idle'` only; no "generating" flash while answer content streams

**Validation (must pass after rebuild/restart)**
1. "can you give me 5 animals with the letter p?" → Real list of 5 P-animals
2. "name 5 girls names with the letter g?" → Real list of 5 G-girl names
3. "list 5 programming languages that start with P?" → Real list of 5 P-languages

No refusals, no "rephrase your question", no therapy/coaching language.

---

## Part 2: Local LLM via llama.cpp — OPTION B (Selected)

**Decision:** Local model via llama.cpp/ONNX Runtime (Option B).

**Recommended model:** `Llama-3.1-8B-Instruct-Q4_K_M.gguf` (~4.7 GB)
**Alternative:** `Llama-3.2-3B-Instruct-Q4_K_M.gguf` (~2.0 GB) — smaller, faster, lower RAM

**Design:**
- `Language::respond()` keeps rule-based fast path (lists, greetings, code examples) for instant deterministic answers
- When rule-based handlers don't match AND web search fails, fall back to the local LLM via llama.cpp C++ API
- `CoreAI::forward()` in `Core.cpp` remains as the 8-class classifier — it is NOT replaced and NOT an LLM
- Model file stored locally (e.g., `models/llama-3.1-8b-instruct-q4_k_m.gguf`)

**Files affected:**
- `CoreAI3D/src/Language.cpp` — add llama.cpp fallback in `respond()`, load model on init
- `CoreAI3D/include/APIServer.hpp` — add LLM config members (`llmModelPath`, `llmContextSize`, `llmTemperature`, `llmMaxTokens`)
- `CoreAI3D/src/APIServer.cpp` — load LLM config from `config.json`, expose model path to Language
- `web/app/api/chat/route.ts` — no changes needed (frontend already passes `model` param; backend handles routing)

**Build dependencies (llama.cpp):**
- Clone llama.cpp or add as submodule
- Build `libllama.a` / `llama.cpp` shared lib
- Link against C++ backend
- Model file must be present at runtime or LLM fallback is skipped gracefully

**Fallback behavior:** If model file is missing or llama.cpp fails to load, `respond()` falls through to existing generic fallback responses (no crash).

---

## Build/Deploy notes (implementation-ready)
- Rebuild C++ backend: compile `CoreAI3D/src/*.cpp` with `g++ -std=c++17` plus MySQL/Boost link flags + llama.cpp
- Download GGUF model to `CoreAI3D/models/llama-3.1-8b-instruct-q4_k_m.gguf` (~4.7 GB)
- Rebuild Next.js frontend on coreai3d.nl
- Restart both services
- Run the 3 validation queries above
