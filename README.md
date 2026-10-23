# Pi Engineering Assistant

A public source-code snapshot of an engineering assistant built around small local
language models, retrieval, deterministic calculators and an inspectable web interface.
The standalone calculator demo runs offline on an ordinary computer, without a model
or a Raspberry Pi.

[Project overview](https://philia17.github.io/research/pi-engineering-assistant/)

A small [browser-only companion calculator](https://philia17.github.io/demos/engineering-calculator/) illustrates formula and calculation traces with generic circuit examples. It is separate from the original research implementation and benchmark.

## What is included

- Deterministic calculators for power converters, transformers, phasors, statistics,
  numerical methods and basic programming calculations.
- Structured answers, calculation traces, citation contracts and input checks.
- Retrieval/indexing code, an Ollama client, and optional vision adapters.
- A FastAPI application and the browser interface for Ask and Quiz workflows.
- Fresh generic examples and an offline test suite.

The public export omits university teaching material, assessment fixtures, research
benchmark packets, model weights, private deployment configuration and manuscript
sources. The subject catalog starts with zero supplied notes; the example-image
collections and course-derived Quiz property catalog are empty.

## Run the offline demo

Python 3.10 or later is required; the commands below were checked with Python 3.13.
From this directory:

```bash
python -m examples.offline_demo
```

This example uses only the standard library. It returns a shared-heatsink result
of **18 W** and **4.5 °C/W**, a **4:1** ideal-transformer turns ratio, and an
**RC time constant of 0.1551 s**. The numerical inputs were newly selected for this
public example and are separate from the research benchmark.

## Offline development checks

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
ruff check .
ruff format --check .
```

A fresh virtual environment successfully installed the development requirements.
The **18 tests** cover numerical calculations, invalid transformer inputs, phasor
cancellation, serializable answer contracts, concealed Quiz answers before submission,
and the absent optional research-automation module. No model, cloud service or hardware
was contacted. The dependency file is a practical development set, not a locked environment.

## Application and retrieval source

`web_ui/api.py` and `web_ui/integration_v1/` retain the main application code.
`requirements.txt` lists the broader application dependencies. Full Ask/Quiz use also
requires an owner-supplied, redistribution-authorized corpus, locally built indexes,
and a compatible local model server. These inputs and the full model-backed workflow
were not installed or validated for this source export.

The eight subject slots in `rag/SUBJECT_CATALOG.json` are empty. To integrate your own
material, follow the metadata shape in
[`rag/SUBJECT_NOTE_TEMPLATE.md`](rag/SUBJECT_NOTE_TEMPLATE.md), update the catalog
counts and review status, then use the included note-validation and index-building
scripts. Existing index/review checks are retained; this release does not bypass them
or ship fabricated research receipts. Course-derived deterministic Quiz mappings and
private reference-image routes have been removed.

For direct numerical use, import a calculator and pass explicit inputs:

```python
from src.tools.pe_calculators import calculate_rc_time_constant

result = calculate_rc_time_constant(resistance_ohm=3300.0, capacitance_f=47e-6)
print(result.values["time_constant_s"])
```

The legacy item-ID adapter and its embedded assessment values were removed. General
formula functions remain available through their explicit parameter interfaces.

## Research context and contribution

The underlying team research is associated with Paper 106,
*“A Lightweight Test and Training Platform Utilizing a Raspberry Pi 5:
AI-retrieval, Tools and a Compact LLM,”* accepted for oral presentation at IHCI 2026.
Acceptance is the confirmed status; proceedings publication and indexing are not
claimed here.

Siheng Ye's recorded work includes deploying the quantized local model, integrating
course-material retrieval and deterministic tools, designing the inspectable web
workflow, and comparing system configurations. The reported seven-configuration,
96-question quality evaluation used an **offline GPU reference**. Separate Raspberry
Pi 5 CPU observations used the same quantized model artifact. This public snapshot
contains neither that restricted benchmark nor a claim that its results were rerun.

## Source and reuse

The source snapshot is based on commit
`f96e5efa74956062cb47a8b88439a540b1547de6`, with release-only removal of private inputs,
portable catalog setup, empty asset routes, and fresh examples/tests. It contains no
Git history from the research repository.

See [RIGHTS.md](RIGHTS.md) for reuse boundaries and retained third-party notices.
No blanket open-source license has been assigned to the project code. Public visibility
alone does not grant an additional software license.
