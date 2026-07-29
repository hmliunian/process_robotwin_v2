# Project Cleanup Plan

## Current Issues

### 1. Root directory pollution (14 markdown files)
Many temporary/duplicate documentation files from development process.

### 2. Missing files from architecture design
Key components defined in section 11 but not created.

### 3. Unplanned directories
`scripts/`, `tools/`, `run/`, `docs/` - need to decide keep/reorganize/delete.

---

## Proposed Actions

### A. Documentation Cleanup

#### Keep (3 files):
- ✅ `README.md` - Main entry point
- ✅ `EXECUTIVE_SUMMARY.md` - Quick reference for stakeholders
- ✅ `process_data_v2_architecture_design.md` - Detailed design spec

#### Move to docs/ (if needed):
- `QUICKSTART.md` → `docs/QUICKSTART.md`
- `docs/qwen_limitation_and_improvements.md` - already there, keep

#### Delete (development artifacts):
- ❌ `DELIVERY_CHECKLIST.md`
- ❌ `DOCS_INDEX.md`
- ❌ `EXECUTION_TASK_FINAL.md`
- ❌ `FINAL_SUMMARY.md`
- ❌ `NEXT_STEPS.md`
- ❌ `PROGRESS.md`
- ❌ `PROJECT_COMPLETE.md`
- ❌ `PROJECT_SUMMARY.md`
- ❌ `STATUS.md`
- ❌ `TOOL_TASK_COMPLETE.md`

### B. Create Missing Core Files (stubs with TODOs)

Following section 11 of architecture design:

```
src/robotwin_annotation_v2/
  application/
    ✅ prepare_keyframes.py (exists)
    ✅ review_keyframes.py (exists)
    🆕 dto.py
  
  adapters/
    🆕 robotwin_dataset.py
    🆕 qwen_grounding.py
    🆕 sam3_single_frame.py
    🆕 filesystem_artifacts.py
    🆕 image_rendering.py
    🆕 human_review.py
    ✅ fake_adapters.py (for testing, keep)
  
  bootstrap/
    🆕 container.py
    🆕 settings.py
  
  cli/
    🆕 keyframes.py

tests/
  🆕 fixtures/
```

### C. Organize New Directories

#### scripts/ - Keep but document
- Purpose: Development/deployment helper scripts
- Keep: `serve_qwen_v2.py`, `restart_qwen_v2.sh`, etc.
- Add: `scripts/README.md` to explain purpose

#### tools/ - Keep but rename?
- Current: `test_data_one_task.py`, `check_status.sh`
- Option 1: Keep as `tools/` for one-off utilities
- Option 2: Move to `scripts/` if they're operational scripts
- Recommend: Keep separate, add `tools/README.md`

#### run/ - Should be in .gitignore
- Contains: `qwen_v2/server.log`
- Action: Add `run/` to `.gitignore`, document in README where logs go

#### artifacts/data_one_task_viz/ - Unexpected
- Contains: Test visualization outputs
- Expected: artifacts/ should have structure from section 9
- Action: Move to `tests/fixtures/` or delete if regenerable

### D. Directory Structure Target

```
process_data_v2/
├── README.md                          # Main entry, links to other docs
├── EXECUTIVE_SUMMARY.md               # Quick reference
├── process_data_v2_architecture_design.md  # Full spec
├── pyproject.toml
├── justfile
│
├── docs/                              # All supplementary docs
│   ├── QUICKSTART.md
│   └── qwen_limitation_and_improvements.md
│
├── configs/
│   ├── pilot_move_pillbottle_pad.yaml
│   └── data_one_task.yaml
│
├── src/robotwin_annotation_v2/
│   ├── domain/                        # ✅ Complete
│   ├── application/                   # ⚠️ Missing dto.py
│   ├── ports/                         # ✅ Complete
│   ├── adapters/                      # ⚠️ Missing 6 files
│   ├── bootstrap/                     # ⚠️ Missing 2 files
│   └── cli/                           # ⚠️ Missing keyframes.py
│
├── tests/
│   ├── unit/                          # ✅ Has tests
│   ├── contract/                      # ⚠️ Empty
│   ├── integration/                   # ⚠️ Empty
│   └── fixtures/                      # 🆕 Need to create
│
├── scripts/                           # Development scripts
│   ├── README.md                      # 🆕 Document purpose
│   ├── serve_qwen_v2.py
│   └── ...
│
├── tools/                             # One-off utilities
│   ├── README.md                      # 🆕 Document purpose
│   └── test_data_one_task.py
│
├── run/                               # Runtime logs (gitignored)
└── artifacts/                         # Runtime outputs (gitignored)
    └── keyframes/                     # Per section 9
        └── runs/<run_id>/
```

---

## Execution Steps

1. **Delete temporary docs** (10 files)
2. **Move docs to docs/** (QUICKSTART.md)
3. **Create missing stub files** with TODO comments
4. **Add README.md in scripts/ and tools/**
5. **Update .gitignore** to include `run/`
6. **Clean artifacts/** directory
7. **Update main README.md** to reflect new structure
8. **Commit cleanup** as single atomic change

---

## Questions to Resolve

1. Keep `data_one_task.yaml` config? (Not in original design)
2. Keep `tools/test_data_one_task.py`? (What's its purpose?)
3. What's in `artifacts/data_one_task_viz/`? Test fixtures or real outputs?

---

**Estimated Time**: 30-45 minutes
**Risk**: Low (mostly file operations, no code logic changes)
**Benefit**: Project matches design spec, easier onboarding
