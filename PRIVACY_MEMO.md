# Privacy memo

One page. Due with the project, Fri Nov 20. Four points of your final grade,
which is more than any single level of the code.

Replace this file with your memo. Keep the four headings; delete everything
else.

---

## The task

You built something that finds faces. Write the memo you would owe someone
before it got switched on.

Not a summary of what you built: the code is in the same repo and we can read
it. The memo answers a different question: what does this thing do to the
people it is pointed at, and what did you actually establish about it?

One page means one page. Roughly 500 words: short enough that the person who
needs it will actually read the whole thing.

## The four sections

**1. What it does, in plain language.** Someone who does not write code has to
understand what your app does, what it stores, and what it does not do. If the
distinction between detection and recognition matters here (and it does)
this is where you make it, without using either word as jargon.

**2. Where the data came from.** Read `provided/PROVENANCE.md`. Say where the
faces came from, who consented to what, and what that means for anyone
deploying this on real people. If you think there is a problem with the answer,
write that down; there is at least one, and finding it is part of the exercise.

**3. What you measured, and what you did not.** Report your L5 numbers. Then
state plainly what those numbers do and do not establish about this system.

This is the section that separates the memos. The corpus supports one kind of
gap and not another, `PROVENANCE.md` says which, and the difference is the
whole point. A team that measures the gap they can measure, says clearly that
it is not the gap in the literature, and cites the literature for that one will
score better than a team that blurs the two. Confident claims your own data
cannot support cost more than acknowledged limitations.

One specific trap, because the faces look real enough to walk into it. Nothing
stops you from sorting these images by apparent skin tone yourself and
computing a detection rate for each pile. You will get a number. That number
does not mean what it appears to mean, and `PROVENANCE.md` explains why in one
paragraph: these faces came out of a generator trained on photographs, so any
difference you find is inseparable into "the detector handles these faces
worse" and "the generator draws these faces worse". Reporting it as a finding
about the detector is the single most expensive mistake available in this
memo. Noticing that you could do it, doing it, and then explaining why you are
not reporting it as a result would be an excellent paragraph.

**4. What you would want in place before this ran on real people.** Concrete
and specific. "Be ethical" is not a control. Who can turn it on, who sees the
output, how long anything is kept, what happens when it fails on someone, and
who they complain to.

## Rubric: 4 points

| | Points |
|---|---:|
| Plain-language account a non-programmer could follow; detection and recognition kept distinct | 1.0 |
| Provenance engaged honestly, including what the consent question actually costs | 1.0 |
| Measured claims separated from cited claims; limitations stated rather than implied | 1.5 |
| Controls that are specific enough to implement | 0.5 |

Half marks exist. Over a page loses 0.5. A memo that is mostly a description of
your code, however good the code, tops out around 1.5.

### On the 1.5-point section

It is weighted highest on purpose, and it is the one where a fluent memo can
still score badly. Four ways teams lose it:

- Reporting the per-group gap and letting the reader assume it is the
  demographic result they have heard about.
- Citing the demographic literature and implying your measurement confirms it.
- Running your own demographic split on this corpus and reporting the result
  as a property of the detector.
- Reporting nothing but the numbers, with no statement of scope at all.

Getting it right does not require hedging everything. It requires one clear
sentence about which claim rests on your data and which rests on somebody
else's.

