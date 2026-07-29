# RoboTwin Annotation V2

Keyframe-first annotation system for RoboTwin manipulation videos.

## Architecture

Clean architecture with phase isolation:
- **Phase 1** (current): Single-frame keyframe seeds (target, receiver)
- **Phase 2** (future): Video propagation
- **Phase 3** (future): Full video QC

## Quick Start

```bash
# Install phase 1 dependencies
just install

# Check GPU availability
just check-gpu

# Run tests
just test-fast

# Prepare keyframes for an episode
just prepare-keyframes 007152

# Run full pilot (10 episodes)
just run-pilot
```

## Project Structure

```
process_data_v2/
├── src/robotwin_annotation_v2/
│   ├── domain/          # Business entities, policies, rules
│   ├── application/     # Use cases (PrepareKeyframes, ReviewKeyframes)
│   ├── ports/           # External capability interfaces (Protocol)
│   ├── adapters/        # Concrete implementations (SAM3, Qwen, RoboTwin)
│   ├── bootstrap/       # Dependency injection
│   └── cli/             # Command-line interface
├── tests/
│   ├── unit/            # Fast tests, no real models
│   ├── contract/        # Interface contract tests
│   └── integration/     # Full pipeline tests with real models
├── configs/             # YAML configurations
└── artifacts/           # Runtime outputs (gitignored)
```

## Design Principles

1. **Keyframe first, propagation second**: Only approved seeds can be propagated
2. **Immutable artifacts**: Every run has `run_id`, versioned, traceable
3. **Explicit failure**: No silent fallbacks; rejected = explicitly marked
4. **Phase isolation**: Phase 1 never calls video propagation
5. **Clean dependencies**: Domain → Application → Ports → Adapters

## Reuse from process_data

- **Data**: Reads from `../process_data/data/`
- **Qwen service**: HTTP client to existing `serve_qwen.py` on port 18086
- **Checkpoints**: Shares SAM3/CoTracker weights

## Commands

See `just --list` for all available commands.

Key workflows:
- `just install` - Set up environment
- `just test` - Run test suite
- `just prepare-keyframes <episode_id>` - Generate keyframe candidates
- `just review-keyframes <run_id>` - Review and approve seeds
- `just run-pilot` - Process all pilot episodes

## Development

```bash
# Type checking
just typecheck

# Linting
just lint

# Format code
just fmt

# Interactive shell
just shell
```

## Phase 1 Deliverable

For each episode:
- ✅ Traceable keyframe candidates (target_0, receiver_0)
- ✅ QC reports (geometry + semantic)
- ✅ Human-reviewable contact sheets
- ✅ Approved seeds with full provenance
- ❌ No video propagation (reserved for Phase 2)
- ❌ No full-video masks.npz (reserved for Phase 2)
