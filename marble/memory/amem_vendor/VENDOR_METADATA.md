# Vendored A-MEM Metadata

- Upstream repository: https://github.com/agiresearch/A-mem
- Upstream package path: `agentic_memory/`
- Pinned commit: `ceffb86` (HEAD of upstream at vendor time)
- Vendored on: 2026-02-27
- Local path: `marble/memory/amem_vendor/`

## Local Adaptations

- Removed global ChromaDB reset behavior from default initialization.
- Added explicit per-instance collection naming and optional targeted reset controls.
- Trimmed unused imports and optional heavy features that are not used by MARBLE integration.
- Kept the public methods needed by MARBLE adapter (`add_note`, `search_agentic`, `read`, `update`, `delete`).
