# Installing, on any machine

## The short version

```bash
pip install kronikas          # into any Python 3.10-3.12
kronikas skill install        # copy this skill to ~/.claude/skills
```

That is the whole install. The guided workflow is part of the package: the
`guided`, `report` and `skill` subcommands come with it, and no plotting or
YAML library is needed on top.

Expect a few minutes for the `pip install`: PyMC and its compiler toolchain
are large. Warn the user before starting rather than leaving them staring at a
silent terminal.

## When there is no environment yet

```bash
bash $SKILL/scripts/setup_kronikas.sh --dir <workdir>
```

Picks the newest Python between 3.10 and 3.12, creates
`<workdir>/.kronikas-venv`, installs kronikas, verifies the import, and prints
the paths to use afterwards. Re-running it is safe and upgrades in place.

`--source` clones `https://github.com/vtisza/kronikas` into `<workdir>/kronikas`
and installs from that instead. Use it when the user wants the source, an
unreleased change, or a version they can edit.

By hand, the same thing:

```bash
python3.12 -m venv .kronikas-venv
.kronikas-venv/bin/python -m pip install kronikas
.kronikas-venv/bin/kronikas guided forecast.yaml --check
```

**Windows PowerShell:**

```powershell
py -3.12 -m venv .kronikas-venv
.\.kronikas-venv\Scripts\python.exe -m pip install kronikas
.\.kronikas-venv\Scripts\kronikas.exe skill install
.\.kronikas-venv\Scripts\kronikas.exe guided forecast.yaml --check
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

Nothing here is Claude-specific — it is a documented CLI:

```bash
kronikas guided forecast.yaml --check   # validate and read back
kronikas guided forecast.yaml           # fit and report
kronikas report out/report_data.json    # rebuild the page, no refit
```

- **Another assistant** (ChatGPT, Codex, Cursor, Copilot Chat, an agent with
  shell access): run `kronikas skill install --dir <somewhere>` and paste
  `SKILL.md` in as instructions. The workflow, the interview questions and the
  commands are all written to be followed by any capable assistant.
- **No assistant at all**: copy `assets/forecast.template.yaml`, fill it in by
  hand — every setting is commented — and run the commands above.
- **Where the skill goes.** `kronikas skill install` writes to
  `~/.claude/skills/election-forecast`, which loads in every session, in any
  directory. `--dir <project>/.claude/skills` installs it for one project and
  everyone working on it instead. `--force` overwrites an existing copy;
  without it an install that would clobber your edits stops.

## Offline and air-gapped machines

The install step needs the network once. After that nothing does: the model
runs locally and `report.html` embeds its own styling and charts, so it opens
with no connection.
