# Installing, on any machine

## The normal path

```bash
bash $SKILL/scripts/setup_kronikas.sh --dir <workdir>
```

Picks the newest Python between 3.10 and 3.12, creates
`<workdir>/.kronikas-venv`, installs kronikas from PyPI, verifies the import,
and prints the interpreter path to use afterwards. Re-running it is safe and
upgrades in place.

`--source` clones `https://github.com/vtisza/kronikas` into `<workdir>/kronikas`
and installs from that instead. Use it when the user wants the source, an
unreleased change, or a version they can edit.

Expect a few minutes: PyMC and its compiler toolchain are large. Warn the user
before starting rather than leaving them staring at a silent terminal.

## Doing it by hand

Any environment where `import kronikas` works will do. The scripts only need
kronikas and its own dependencies — no plotting library, no YAML library.

```bash
python3.12 -m venv .kronikas-venv
.kronikas-venv/bin/python -m pip install kronikas
```

**Windows PowerShell:**

```powershell
py -3.12 -m venv .kronikas-venv
.\.kronikas-venv\Scripts\python.exe -m pip install kronikas
.\.kronikas-venv\Scripts\python.exe <skill>\scripts\run_forecast.py forecast.yaml --check
```

**conda / mamba** (better BLAS, noticeably faster sampling):

```bash
conda create -n kronikas -c conda-forge python=3.12 pymc
conda run -n kronikas pip install kronikas
```

## Python version

kronikas requires **3.10, 3.11 or 3.12**. On 3.13+ the install fails with a
message about no matching distribution. That is a real constraint from PyMC,
not something to work around — install a supported interpreter:

- macOS: `brew install python@3.12`
- Debian/Ubuntu: `sudo apt install python3.12 python3.12-venv`
- Windows: python.org installer for 3.12, then use `py -3.12`

## Using this outside Claude Code

Nothing here is Claude-specific. The three scripts are ordinary Python and
shell:

```bash
bash scripts/setup_kronikas.sh --dir ~/forecast
~/forecast/.kronikas-venv/bin/python scripts/run_forecast.py forecast.yaml --check
~/forecast/.kronikas-venv/bin/python scripts/run_forecast.py forecast.yaml
```

- **Another assistant** (ChatGPT, Codex, Cursor, Copilot Chat, an agent with
  shell access): paste `SKILL.md` in as instructions and point it at this
  directory. The workflow, the interview questions and the commands are all
  written to be followed by any capable assistant.
- **No assistant at all**: copy `assets/forecast.template.yaml`, fill it in by
  hand — every setting is commented — and run the two commands above.
- **Installing the skill for every project** in Claude Code: copy the
  `election-forecast` directory into `~/.claude/skills/`. It then loads in any
  session, in any directory. Copy it into a project's `.claude/skills/`
  instead to make it available to everyone working on that project.

## Offline and air-gapped machines

The install step needs the network once. After that nothing does: the model
runs locally and `report.html` embeds its own styling and charts, so it opens
with no connection.
