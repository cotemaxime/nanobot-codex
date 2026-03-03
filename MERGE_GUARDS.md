# Merge Guards for Fork-Kept Features

This fork intentionally keeps behavior differences from upstream.

Use this file as a merge checklist when syncing from HKUDS/nanobot.

## Kept features and test coverage

### 1) Topic-scoped session behavior (Telegram threads)
- Feature: topic/thread messages map to stable session keys, and per-topic state stays isolated.
- Covered by:
  - `tests/test_merge_guard_kept_features.py::test_model_override_is_scoped_by_topic_session_key`
  - `tests/test_merge_guard_kept_features.py::test_skill_selection_is_scoped_by_topic_session_key`
  - `tests/test_message_bus.py::test_has_pending_inbound_for_session_uses_metadata_session_key`
  - `tests/test_telegram_reaction_registration.py::test_reaction_metadata_uses_tracked_thread_id`
  - `tests/test_telegram_reaction_registration.py::test_topic_command_roundtrip_sends_reply_to_same_thread`

### 2) Slash command UX kept in chat loop
- Feature: command handling remains available from chat flow (`/new`, `/help`, `/last`, `/skills`, `/skill`, `/model`).
- Feature detail: `/new` rotates the current session JSONL into `~/.nanobot/sessions/archives/` before resetting active session history.
- Covered by:
  - `tests/test_agent_loop_codex_parity.py::test_new_command_with_bot_mention_resets_without_model_call`
  - `tests/test_agent_loop_codex_parity.py::test_last_command_resends_previous_assistant_message`
  - `tests/test_merge_guard_kept_features.py::test_help_command_lists_kept_chat_commands`
  - `tests/test_merge_guard_kept_features.py::test_model_override_is_scoped_by_topic_session_key`
  - `tests/test_consolidate_offset.py::TestSessionArchiveOnReset::test_reset_moves_previous_non_empty_session_to_archives`
  - `tests/test_slash_commands.py::test_slash_loader_parses_frontmatter_and_body`

### 3) Reaction handling workflow (Telegram)
- Feature: reactions trigger approval/redo/retry behavior and compatible handler registration.
- Covered by:
  - `tests/test_agent_loop_codex_parity.py::test_telegram_reaction_thumbs_up_marks_completed`
  - `tests/test_agent_loop_codex_parity.py::test_telegram_reaction_redo_resends_last_assistant`
  - `tests/test_agent_loop_codex_parity.py::test_telegram_reaction_retry_replays_last_user_request`
  - `tests/test_telegram_reaction_registration.py::test_register_reaction_handlers_uses_filter_constants`
  - `tests/test_telegram_reaction_registration.py::test_register_reaction_handlers_falls_back_to_type_handler`

### 5) Subagent behavior and UX
- Feature: immediate background notice, heartbeat progress, capped heartbeat backoff, and codex fallback behavior.
- Covered by:
  - `tests/test_subagent_manager.py::test_spawn_sends_immediate_background_notice`
  - `tests/test_subagent_manager.py::test_spawn_emits_heartbeat_progress`
  - `tests/test_subagent_manager.py::test_heartbeat_delay_schedule_caps_at_ten_minutes`

### 8) Disabled-skills enforcement and config migration
- Feature: disabled skills are enforced at load/list time, and legacy config keys are migrated safely.
- Covered by:
  - `tests/test_skills_loader.py::test_disabled_skills_are_hidden_from_list`
  - `tests/test_skills_loader.py::test_disabled_skill_cannot_be_loaded`
  - `tests/test_config_loader.py::test_migrate_moves_disabled_skills_from_agents_defaults`
  - `tests/test_config_loader.py::test_migrate_does_not_override_existing_agents_disabled_skills`
  - `tests/test_filesystem_tool_security.py::test_read_file_blocks_disabled_skill_paths`
  - `tests/test_filesystem_tool_security.py::test_list_dir_blocks_disabled_skill_directory`

### 9) Context limits and compaction warnings
- Feature: separate main/cron context limits and compaction summary flow should remain intact.
- Covered by:
  - `tests/test_context_limits_config.py::test_build_context_limits_main_and_cron_models`
  - `tests/test_context_limits_config.py::test_build_context_limits_defaults_cron_to_main_limit`
  - `tests/test_context_compaction.py::test_compact_command_reduces_session_messages`
  - `tests/test_context_compaction.py::test_context_warning_sets_pending_compact_action`

### 10) Concurrency and queue behavior
- Feature: parallel sessions should not block each other, and per-channel outbound workers stay independent.
- Covered by:
  - `tests/test_agent_loop_codex_parity.py::test_agent_run_processes_sessions_concurrently`
  - `tests/test_merge_guard_kept_features.py::test_channel_manager_uses_per_channel_outbound_workers`

## Fork additions/options to preserve (high-level)

- Added command surfaces and UX:
  - Workspace slash commands and topic-safe command routing.
  - Session archive-on-reset behavior for `/new`.

