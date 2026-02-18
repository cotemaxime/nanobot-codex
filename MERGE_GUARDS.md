# Merge Guards for Fork-Kept Features

This fork intentionally keeps a small set of behavior differences when syncing from upstream.

Use this file as a checklist during each upstream merge.

## Kept features and test coverage

### 1) Topic-scoped session behavior (Telegram threads)
- Feature: topic/thread messages map to stable session keys, and per-topic state stays isolated.
- Covered by:
  - `tests/test_merge_guard_kept_features.py::test_model_override_is_scoped_by_topic_session_key`
  - `tests/test_merge_guard_kept_features.py::test_skill_selection_is_scoped_by_topic_session_key`
  - `tests/test_message_bus.py::test_has_pending_inbound_for_session_uses_metadata_session_key`

### 2) Slash command UX kept in chat loop
- Feature: command handling remains available from chat flow (`/new`, `/help`, `/last`, `/skills`, `/skill`, `/model`).
- Covered by:
  - `tests/test_agent_loop_codex_parity.py::test_new_command_with_bot_mention_resets_without_model_call`
  - `tests/test_agent_loop_codex_parity.py::test_last_command_resends_previous_assistant_message`
  - `tests/test_merge_guard_kept_features.py::test_help_command_lists_kept_chat_commands`
  - `tests/test_merge_guard_kept_features.py::test_model_override_is_scoped_by_topic_session_key`

### 3) Reaction handling workflow (Telegram)
- Feature: reactions trigger approval/redo/retry behavior.
- Covered by:
  - `tests/test_agent_loop_codex_parity.py::test_telegram_reaction_thumbs_up_marks_completed`
  - `tests/test_agent_loop_codex_parity.py::test_telegram_reaction_redo_resends_last_assistant`
  - `tests/test_agent_loop_codex_parity.py::test_telegram_reaction_retry_replays_last_user_request`

### 4) Context-aware tool routing
- Feature: tools fall back to current runtime route when explicit destination is not provided.
- Covered by:
  - `tests/test_merge_guard_kept_features.py::test_message_tool_uses_runtime_context_when_no_explicit_target`
  - `tests/test_merge_guard_kept_features.py::test_spawn_tool_uses_runtime_context_when_no_explicit_target`
  - `tests/test_merge_guard_kept_features.py::test_cron_tool_uses_runtime_context_when_no_explicit_target`

### 5) Subagent behavior changes
- Feature: immediate background notice and delayed completion reporting format.
- Covered by:
  - `tests/test_subagent_manager.py::test_spawn_sends_immediate_background_notice`

### 6) Concurrency and queue behavior
- Feature: parallel sessions should not block each other, and per-channel outbound workers stay independent.
- Covered by:
  - `tests/test_agent_loop_codex_parity.py::test_agent_run_processes_sessions_concurrently`
  - `tests/test_merge_guard_kept_features.py::test_channel_manager_uses_per_channel_outbound_workers`

## How to run merge-guard checks

Run full suite:

```bash
.venv/bin/python3 -m pytest -q
```

Run only merge-guard focused tests:

```bash
.venv/bin/python3 -m pytest -q \
  tests/test_merge_guard_kept_features.py \
  tests/test_agent_loop_codex_parity.py \
  tests/test_message_bus.py \
  tests/test_subagent_manager.py
```

## Upstream sync checklist

1. Rebase/cherry-pick onto upstream head.
2. Resolve conflicts in fork-kept files first.
3. Run merge-guard focused tests.
4. Run full test suite.
5. If any guard fails, fix behavior or explicitly update this file if feature intent changed.
