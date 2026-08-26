# Gold Set Labeling Guidelines

You're judging extracted action items (from both the LLM and classifier arms) against the source transcript. Use `scripts/labeling_app.py` (`streamlit run scripts/labeling_app.py`) rather than editing the CSV directly.

## Assisted labeling (suggestions)

Each row shows a **suggested** task/owner/deadline judgment with a plain-English reason (e.g. "contains a question, not a clean commitment"). This comes from simple pattern-matching (`confidence/pseudo_label_heuristic.py`) - modal verbs, question marks, whether the evidence quote is actually grounded in the transcript, etc. - **not from either comparison arm's own model**, specifically to avoid the circularity of an LLM pre-judging its own output.

The suggestion is a starting point, not the answer - it's wrong often enough that you need to actually read the evidence each time, not just click accept reflexively. Use "✓ Accept suggestion" when you agree, or answer manually when you don't.

**Honest limitation**: even with real review, pre-filled suggestions can anchor judgment somewhat compared to labeling fully cold - worth naming in the final write-up as a caveat on the gold set's independence, not something to hide.

For each row, the underlying judgment is three questions, each `y`/`n`:

## `correct_task` — is this a genuine action item, described accurately?

- `y`: the transcript really does contain a concrete commitment or assigned task, and `text_shown` describes it accurately (paraphrasing is fine — wording doesn't have to match exactly, the *content* does).
- `n`: this is general discussion, an opinion, a question, a filler sentence — not an actual action item. Also `n` if it IS a real action item but the description is wrong/garbled (e.g. merges two unrelated things, or a MeetingBank sentence-splitting artifact cut it off mid-thought).

Don't judge importance here — a low-priority action item is still `y` if it's real.

## `correct_owner` — is the claimed owner right?

- `y` if `owner` names the right person **and** the transcript actually supports that. Also `y` if `owner` is blank AND the transcript genuinely doesn't make the owner clear (correctly recognizing ambiguity is correct, not a miss).
- `n` if `owner` names the wrong person, or is blank when the transcript actually does make the owner clear.

## `correct_deadline` — is the claimed deadline right?

- `y` if `deadline` matches what's stated, or is blank when no deadline was actually mentioned.
- `n` if `deadline` is wrong, invented, or blank when a deadline really was stated.

## Mechanics

- `evidence_span` is the quote the row is based on — check it against the transcript. For the `classifier` arm, `evidence_span` and `text_shown` are the same (the raw candidate sentence); for the `llm` arm, `text_shown` is a paraphrased task while `evidence_span` is the supporting quote.
- If `evidence_span` looks garbled or doesn't actually appear in the transcript, that's already useful signal — mark `correct_task` = `n` and note it in `notes`.
- Full transcripts are in `data/processed/<qmsum|meetingbank>/<meeting_id>.json` if you need more context than the evidence_span gives you.
- Leave `notes` for anything worth flagging (ambiguous call, transcript itself is confusing, sentence-splitting artifact, etc.) — useful later, not required.

## A note on consistency

Since this is single-annotator labeling (no second labeler to check agreement against), do a short stability check after finishing: re-label a small random sample (~10-15 rows) a day or so later without looking at your first pass, and compare. Large disagreement with yourself means the rubric above needs tightening before trusting the calibration model trained on these labels.
