# Team Project: The Detector, and What It Misses

Four people, three weeks, one small Python package. Out Mon Nov 2, due Fri
Nov 20. This file is the assignment; there is no separate handout.

`DATA → MODEL → TRAIN → EVALUATE → DEPLOY`: this project lives at **DEPLOY**.
You will not train anything. Both detectors already exist, and the work is
using them correctly and knowing where they fail.

You have spent the semester in notebooks. This is a repo. The difference is not
cosmetic: a notebook remembers everything you ever ran, in the order you ran it,
and a module imported into a fresh interpreter remembers nothing. That is why
the house rule changes shape here. Instead of Restart & Run All, the rule is
that `import facedetect` has to work from a clean interpreter, and the grader
does exactly that before it calls anything you wrote.

## What you are building

A face-aware application, in five steps, on top of two detectors that already
exist and that you will not train.

| Level | Module | What it does | Points |
|-------|--------|--------------|--------|
| L1 | `facedetect/detect.py` | Run YOLOv8 over the provided scenes; write `detections.csv` | 12 |
| L2 | `facedetect/boxes.py` | IoU and filtering: the arithmetic underneath every detector | 18 |
| L3 | `facedetect/faces.py` | Run BlazeFace; find and count faces | 14 |
| L4 | `facedetect/app.py` | Build the app: blur, count, or overlay | 22 |
| L5 | `facedetect/audit.py` | Measure the detection rate per group | 18 |
| REPO | - | The package imports clean and is organised | 8 |
| ART | - | Your annotated output image | 8 |
| Bonus | `facedetect/boxes.py` | Non-max suppression, by hand, graded on speed too | +10 |

Plus, submitted separately and graded by a human: a one-page privacy memo and a
two-minute demo video.

L4 is the open-ended one. Three modes are offered and the grader accepts any of
them, because it checks that your app does something real to the right part of
the image rather than checking that you built the version we imagined.

## Getting started

**Check your Python version first.** This project needs **Python 3.12**.
`mediapipe` has no wheel for 3.13 or newer, and the failure message you get
does not say so. `python -VV` tells you what you have; Colab is already on a
supported version.

```
pip install -r requirements.txt
python -c "import ultralytics; ultralytics.settings.update({'sync': False})"
python grader.py .
```

The second line turns off Ultralytics' usage reporting, which is on by default.
You are about to write a privacy memo. Consider that line part of the
assignment rather than boilerplate, and notice that nothing forced you to run
it.

The grader on a fresh clone reports zero and tells you which level to start on.
That is the expected first result, not a broken install.

## What ships with the package

- `provided/scenes/` and `scenes.csv`: 24 scenes with ground-truth boxes
- `provided/faces/` and `faces.csv`: 550 face images, each labelled with a
  `group` and a ground-truth box
- `provided/models/`: both sets of weights, loaded by path. Nothing in this
  package downloads anything.
- `provided/PROVENANCE.md`: where the data came from. **Read this before L5.**
  It is short, and one section of it decides whether your memo is right or
  confidently wrong.

The faces are generated, not photographed. No person is depicted, so there was
nobody to ask for consent. That is not a shortcut: it is the only option that
survives contact with how face corpora are actually licensed, and
`PROVENANCE.md` explains both what that buys us and what it costs.

Teams do not collect faces. Not from classmates, not from the internet, not
from your own camera roll. If you want your own face in the demo video, point a
webcam at yourself live; do not add it to the dataset.

## Ground truth is not the detector's output

`faces.csv` and `scenes.csv` record where each object was **placed**. The
detector's opinion is a separate thing, and the gap between the two is the
entire subject of L5. `faces.csv` also carries `n_faces`: it is there so you
can check your work, and reading the answer out of it instead of running a
detector will fail L3, which grades boxes and not counts.

## Submitting

Push to your team repo and upload a zip to D2L containing:

- the `facedetect/` package
- `detections.csv` and your annotated `artifact.png`
- `results.json` from your own grader run
- `PRIVACY_MEMO.md`: one page
- a link to your two-minute demo video

Do not include `provided/` in the zip; we have it. Do not include
`__pycache__/`, and do not include a copy of `grader.py` or `grader_core.py`,
the re-grade uses ours, and a modified copy in your submission is not a
gradable state of affairs.

One person submits per team. The deadline is hard: this lands before break, and
there are no grace days on it.

## How this is graded

The same two-stage arrangement as every other assignment. `python grader.py .`
is a preview you can run as often as you like. The grade of record comes from
re-running the same grader over your submission on our machines, under the
pinned versions in `requirements.txt`. The two agree as long as a clean import
genuinely works on your end.

Partial credit is real and tiered. A function that returns the right shape and
type earns something with the values wrong. A function you never wrote earns
nothing. The grader's messages name which tier you missed; read them before
office hours and bring them with you.

L3 through L5 depend on the levels before them, so a broken L1 will cost you
more than twelve points. Build in order.

## The memo

One page. Not a summary of what you built: we can see what you built. The
prompt, the rubric, and the length limit are in `PRIVACY_MEMO.md`.

The one thing worth saying here: the memo is where you write down what your own
measurements do and do not establish. A team that measures the gap this data
can show, states plainly that it is not the gap in the literature, and cites the
literature for that one, will score better than a team that quietly implies it
measured something it did not.
