You are analyzing a meeting transcript excerpt to extract a summary, key decisions, and action items.

For each action item, you must fill in:

- **task**: what needs to be done, in a short phrase.
- **owner**: the person responsible, if the transcript indicates one. Use `null` if genuinely unclear.
- **owner_attribution**: how confidently the owner was identified, one of:
  - `explicit_self`: the speaker commits to it themselves ("I'll send it")
  - `explicit_named`: someone is directly addressed and accepts/is assigned the task
  - `explicit_third_person_named`: a specific person is named as the owner, but not directly addressed in this excerpt
  - `inferred_pronoun`: a pronoun refers to the owner with no clear antecedent nearby
  - `inferred_unassigned`: no owner language at all; you are guessing based on role or context
- **deadline**: any date/timeframe mentioned, or `null` if none.
- **priority**: `high`, `medium`, or `low`, based on urgency language in the transcript (not your own judgment of importance).
- **evidence_span**: a VERBATIM quote copied exactly from the transcript that supports this action item. Do not paraphrase this field — it must be an exact substring of the transcript text you were given.
- **llm_confidence**: your own honest confidence (`high`/`medium`/`low`) that this action item is correctly extracted (right task, right owner, right deadline).
- **basis**: `explicit_statement` if directly stated, `contextual_inference` if you inferred it from surrounding context, or `speaker_role_default` if you guessed based on someone's apparent role.

Only extract genuine action items — concrete commitments or assigned tasks, not general discussion topics. If there are no action items in this excerpt, return an empty list. Do not invent action items, owners, or deadlines that aren't supported by the text.

**evidence_span is required and must never be empty.** Every action item came from somewhere in the transcript — find the specific sentence or phrase and copy it exactly. If you cannot find a supporting quote, do not include that action item at all.

Example. Given this excerpt:

> Sarah: Okay, so you do the looking around at other remote controls.
> Mike: Sure, I can have that done by next week.

A correctly-filled action item:

```json
{
  "task": "Look around at other remote controls (competitor research)",
  "evidence_span": "so you do the looking around at other remote controls",
  "owner": "Mike",
  "owner_attribution": "explicit_named",
  "deadline": "next week",
  "priority": "medium",
  "llm_confidence": "high",
  "basis": "explicit_statement"
}
```

Note how `owner` and `owner_attribution` agree with each other, and `evidence_span` is copied verbatim from the transcript, not paraphrased or left blank.
