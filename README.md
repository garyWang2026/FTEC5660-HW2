# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution:

### Task 1

My solution is a hybrid pipeline that keeps every step checkable. For each CV I
(1) ask DeepSeek (`deepseek-v4-flash`, via `ChatDeepSeek`) to extract the
candidate's structured claims — name, city, headline, jobs, education, skills —
from the markdown text produced by the same PDF loader used for grading; (2)
resolve the real person with the MCP server: `search_linkedin_people` with the
bare city name and the industry derived from the headline ("X Professional" →
X), because homonyms dominate the mock data — a plain name search returns up to
20 same-name people; (3) if several candidates survive the city/industry
filter, I pull each `get_linkedin_profile` and score company/school overlap
against the CV to pick the right person (this is what distinguishes e.g. the
Munich "AI Professional" Michael Cheung from his same-name distractors); (4)
ask the model to judge differences between the CV claims and the profile across
the seven verifiable fields (name, city, jobs, education, skills), with
explicit rules that wording differences (BSc vs Bachelor of Science, UI/UX
Design vs UI/UX) are not substantive; and (5) map the judged differences to a
deterministic score — no substantive difference → 0.95, otherwise a penalty
that keeps the score ≤ 0.5. Per-CV errors are caught and return a conservative
0.3 so one failure cannot crash the run, and `asyncio.Semaphore(3)` respects
the shared server. On `public_test` the agent scores **7/7 correct** (4 valid,
3 with discrepancies — the shifted Innovate HK Ltd start year, the inflated
Microsoft title, and the invented Docker skill are all caught), and a full run
on the 7 public CVs takes about 75 seconds, well within the 30-minute limit.

### Task 2

My attack is a "real core, fake periphery" CV. All seven fields that verifiers
actually check — name, city, jobs (company/title/years), education and skills —
are copied exactly from Kelly Tsang's true profile (pulled via
`get_linkedin_profile(10001)` and `get_facebook_profile(10001)`), so a verifier
that resolves the right person and checks those fields finds zero substantive
differences and scores above 0.5. The false/embellished details are placed
only in fields the homework explicitly says are never the source of a
discrepancy and that verification coverage does not include: a Professional
Summary claiming a fake "AIA Manager of the Year 2023" award and an 18%
efficiency improvement, a Certifications section listing CFA Level II and FRM
Part I that the profile never mentions, and a Languages section adding
"Japanese (conversational)". This exploits verification coverage (which fields
does a verifier actually check?) rather than prompt injection, which keeps the
CV's visible content honest-looking while still containing real fake
information, as required for Task 2 credit. I tested it against my own Task 1
agent: it resolves person 10001 and scores **0.95 twice** (first draft failed
because I had added fake skills — Power BI/Tableau — into the *skills* field,
which verifiers do check; moving the fakes out of verified fields fixed it).

## Task 3: Reflection on the AI paradoxes

> A short personal reflection. (No prescribed format, so I wrote a short essay.)

**The paradox of verification and trust.** Task 1 is a useful paradox in its
own right: we ask an AI system to judge whether a person's claims are truthful,
yet the judge itself is an AI that can be misled, and the person being judged
may be an AI too (or an AI-assisted human). The verifier's confidence is only
meaningful relative to its blind spots — which Task 2 exploits by hiding lies
exactly where the verifier does not look. The same technology that powers KYC
compliance therefore also powers the attacks against it. This is the
fundamental AI paradox: capability and vulnerability grow together, because a
model that understands human text well enough to verify it also understands it
well enough to be deceived, and a person who can generate convincing content
can now do so at machine scale.

**The paradox of automation in compliance.** KYC exists because humans lie and
make mistakes. Automating it with AI removes human fatigue and scale limits,
but it replaces an imperfect human judge with an imperfect machine judge whose
errors are less transparent and harder to challenge. Automation does not remove
the need for judgement — it moves the judgement to the design of the system:
which fields to check, which differences to ignore, what score threshold means
"trustworthy". The homework makes this visible: the most consequential
decisions (which fields are verifiable, what counts as a discrepancy) are made
by the system designer, not by the AI.

**The paradox of the "same function" adversary.** Task 2 is scored by verifiers
built with the same function as my own Task 1 agent — same model, same tools,
same PDF loader. My attack works against my own verifier, and presumably
against theirs, because the blind spots are shared: any verifier that checks
the seven fields and ignores the summary/certifications/languages will be
fooled by the same trick. This shows that an AI system's weakness is not an
accident of a particular prompt — it is a property of what the designer decided
to look at. "Better" AI in the narrow sense (more accurate extraction, cleaner
reasoning) does not automatically mean safer AI; it can mean a more convincing
attack surface.

**The paradox of scale.** The most unsettling part of the homework is how cheap
the attack was: a few minutes of prompting, and a document that a human
reviewer would flag as embellished passes five AI verifiers. In the past,
fabricating a consistent career history took time and effort; today an LLM can
write one in seconds. The bottleneck is no longer generation — it is
verification, and verification is itself AI-mediated. This is the paradox that
matters for business and FinTech: as AI makes creation nearly free, the value
shifts to trustworthy verification, and that verification cannot be fully
automated because the automated verifier is itself the attack surface. The
practical takeaway is not to trust the score blindly, but to audit the
verifier: what does it check, what could it miss, and who audits the auditor?
