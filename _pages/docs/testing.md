---
permalink: /docs/testing/
title: "Testing"
excerpt: "Running the test suite."
last_modified_at: 2025-12-21T10:00:00-05:00
toc: true
sidebar:
  nav: "docs"
---

## Run all tests (PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
pytest -q
```

## Notes
- Tests live under `tests/`.
- Some tests may mock model providers; configuration is still recommended for local runs.


