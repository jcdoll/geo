# AGENTS.md

> **purpose** – Onboarding guide for humans and AI working on this repository.

---

## 1. Project overview

`geo` is a 2‑D geology simulator. The high‑level description of features and usage lives in [README.md](README.md).  All physical laws and "golden rules" are defined in [PHYSICS.md](PHYSICS.md); consult that document before making physics changes.

---

## 2. Rules

| #   | AI *may* do                                                                       | AI *must NOT* do                                     |
|-----|-----------------------------------------------------------------------------------|------------------------------------------------------|
| R-0 | Ask for clarification when unsure about the project.                              | ❌ Make changes without understanding the context.   |
| R-1 | Edit source files (`*.py`) and documentation when asked.                          | ❌ Delete or rewrite tests unless explicitly asked.  |
| R-2 | Add `AIDEV-NOTE`, `AIDEV-TODO`, or `AIDEV-QUESTION` comments around complex code. | ❌ Remove existing `AIDEV-` comments.                |
| R-3 | Run `pytest` before committing code.                                              | ❌ Ignore failing tests.                             |
| R-4 | Follow PEP8 style (black, 120‑char lines, double quotes).                         | ❌ Reformat code to a different style.               |

---

## 3. Build & test commands

```bash
pip install -r requirements.txt  # install dependencies
pytest                            # run all tests
python main.py                    # start the simulator
```

---

## 4. Key files

| File/Folder              | Description                                   |
|--------------------------|-----------------------------------------------|
| `simulation_engine.py`   | Main simulation engine                        |
| `materials.py`           | Definitions of rock and fluid materials       |
| `atmospheric_processes.py`, `fluid_dynamics.py` | Environmental processes |
| `visualizer.py`          | Pygame-based renderer                         |
| `tests/`                 | Regression tests                              |

---

## 5. Anchor comments

Use `AIDEV-NOTE`, `AIDEV-TODO`, and `AIDEV-QUESTION` to mark important sections of code. Keep them under 120 characters and update existing anchors rather than deleting them.

---

## 6. Directory specific guides

This repository currently contains only this `AGENTS.md`. If you introduce new subdirectories with specialised patterns, add an `AGENTS.md` within them describing those conventions.

---

## 7. Workflow for AI assistants

1. Read this file and [PHYSICS.md](PHYSICS.md) before editing.
2. Ask for clarification if the task is ambiguous.
3. Make the change and include anchor comments when the code is complex.
4. Run `pytest` to ensure tests pass.
5. Commit with a concise message.

