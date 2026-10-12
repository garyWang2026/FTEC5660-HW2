#!/usr/bin/env python3
"""FTEC5660 HW2 student starter: build an agent that verifies CVs via MCP."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


MCP_URL = "https://ftec5660.ngrok.app/mcp"
MODEL_NAME = "deepseek-v4-flash"
THRESHOLD = 0.5


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def cv_files(folder: Path) -> list[Path]:
    """Return PDFs directly inside *folder*, sorted numerically (CV_2 before CV_10)."""

    def key(path: Path) -> tuple[int, str]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return (int(digits) if digits else math.inf, path.name)

    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=key,
    )


def cv_text(path: Path) -> str:
    """Convert one CV PDF to markdown text."""
    from markitdown import MarkItDown

    return MarkItDown(enable_plugins=False).convert(str(path)).text_content


async def load_mcp_tools() -> list[Any]:
    """Connect to the course MCP server and return its tools as LangChain tools."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "social_graph": {
                "transport": "http",
                "url": MCP_URL,
                "headers": {"ngrok-skip-browser-warning": "true"},
            }
        }
    )
    return await client.get_tools()


def build_agent(tools: list[Any]) -> Any:
    """Create and return your agent once.

    ``tools`` are the six SocialGraph MCP tools (Facebook + LinkedIn search and
    profile lookup), already wrapped as LangChain tools. You may add your own
    local tools as well.

    Suggested imports:
        from langchain_deepseek import ChatDeepSeek
        from langchain.agents import create_agent

    Use the DeepSeek model named by ``MODEL_NAME``. The API key is loaded
    from .env.
    """
    from langchain_deepseek import ChatDeepSeek

    llm = ChatDeepSeek(model=MODEL_NAME, temperature=0, max_retries=2, timeout=120)
    return VerificationAgent(llm=llm, tools=tools)


# ---------------------------------------------------------------------------
# VerificationAgent: hybrid design.
#   Deterministic Python steps  -> locate the right person (entity resolution)
#                                  and compute the final score from a diff list.
#   LLM steps                    -> extract structured fields from the CV and
#                                  judge CV-vs-LinkedIn differences (wording
#                                  normalisation is what LLMs do well).
# ---------------------------------------------------------------------------


class VerificationAgent:
    """Wraps the DeepSeek LLM plus the MCP tools for one CV verification run."""

    def __init__(self, llm: Any, tools: list[Any]) -> None:
        self.llm = llm
        self.tools = {t.name: t for t in tools}

    # ---- JSON helpers ------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> Any:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"no JSON object in model output: {text[:200]}")
        return json.loads(text[start : end + 1])

    async def _llm_json(self, system: str, user: str) -> Any:
        from langchain_core.messages import HumanMessage, SystemMessage

        resp = await self.llm.ainvoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        return self._extract_json(str(resp.content))

    # ---- Step 1: extract structured claims from the CV text ---------------

    _EXTRACT_SYSTEM = (
        "You extract structured claims from a CV. Return ONLY a JSON object, no prose.\n"
        'Schema: {"name": str, "city": str, "headline": str, '
        '"jobs": [{"company": str, "title": str, "start_year": int|null, "end_year": int|null}], '
        '"education": [{"degree": str, "school": str, "field": str, "grad_year": int|null}], '
        '"skills": [str]}.\n'
        "Rules:\n"
        "- city = the primary city line (before '|' if present), e.g. 'London, UK'.\n"
        "- headline = the single-line job headline under the name (e.g. 'AI Professional').\n"
        "- jobs: start_year/end_year are integers (end_year null if 'Present').\n"
        "- education: grad_year is the graduation year, degree like BSc/MSc/MBA/PhD, field the major.\n"
        "- skills: the comma-separated skill list, each trimmed.\n"
        "- Transcribe exactly; do not correct or normalise anything."
    )

    async def _extract(self, text: str) -> dict[str, Any]:
        return await self._llm_json(self._EXTRACT_SYSTEM, f"CV text:\n{text}")

    # ---- Step 2: locate the real person (entity resolution) ---------------

    @staticmethod
    def _industry_from_headline(headline: str | None) -> str | None:
        if not headline:
            return None
        h = headline.strip()
        if h.lower().endswith("professional"):
            return h[: -len("professional")].strip()
        return h.strip() or None

    @staticmethod
    def _norm(s: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())

    @staticmethod
    def _match_score(cv: dict[str, Any], profile: dict[str, Any]) -> float:
        """How well does this profile match the CV claims (company/school overlap)."""
        score = 0.0
        cv_jobs = cv.get("jobs") or []
        prof_exp = profile.get("experience") or []
        for job in cv_jobs:
            c = VerificationAgent._norm(job.get("company"))
            if not c:
                continue
            for exp in prof_exp:
                if c and c == VerificationAgent._norm(exp.get("company")):
                    score += 3.0
                    if abs((exp.get("start_year") or 0) - (job.get("start_year") or 0)) <= 1:
                        score += 0.5
                    break
        cv_edu = cv.get("education") or []
        prof_edu = profile.get("education") or []
        for edu in cv_edu:
            s = VerificationAgent._norm(edu.get("school"))
            if not s:
                continue
            for e in prof_edu:
                if s and s == VerificationAgent._norm(e.get("school")):
                    score += 2.0
                    break
        return score

    async def _locate(self, cv: dict[str, Any]) -> dict[str, Any] | None:
        """Return the LinkedIn profile of the person the CV claims to describe."""
        name = (cv.get("name") or "").strip()
        if not name:
            return None
        city = (cv.get("city") or "").strip()
        # The MCP search only accepts a bare city name (no country suffix).
        city_query = city.split(",")[0].strip()
        industry = self._industry_from_headline(cv.get("headline"))
        search = self.tools["search_linkedin_people"]

        def parse_candidates(resp: Any) -> list[dict[str, Any]]:
            if not resp:
                return []
            txt = resp[0]["text"] if isinstance(resp, list) else resp
            cands = json.loads(txt) if isinstance(txt, str) else txt
            if isinstance(cands, dict):
                return [cands]
            return cands if isinstance(cands, list) else []

        def _contains(hay: str | None, needle: str) -> bool:
            h = self._norm(hay)
            n = self._norm(needle)
            return bool(h and n and (n in h or h in n))

        # Primary search: name + city + industry (narrows the homonym pool).
        candidates: list[dict[str, Any]] = []
        for attempt in (
            {"q": name, "location": city_query, "industry": industry, "fuzzy": False},
            {"q": name, "location": city_query, "fuzzy": False},
            {"q": name, "fuzzy": False},
        ):
            try:
                resp = await search.ainvoke(attempt)
            except Exception:
                resp = []
            candidates = parse_candidates(resp)
            if candidates:
                break
        if not candidates:
            return None

        # Pre-filter by city and industry/headline metadata to drop homonyms.
        filtered: list[dict[str, Any]] = []
        for cand in candidates:
            loc_ok = _contains(cand.get("location") or cand.get("city"), city_query)
            head = cand.get("headline") or ""
            ind_ok = (industry is None) or _contains(head, industry) or _contains(cand.get("industry"), industry)
            if loc_ok and ind_ok:
                filtered.append(cand)
        pool = filtered or candidates

        # Score every candidate profile against CV claims; pick the best match.
        best, best_score = None, 0.0
        for cand in pool:
            pid = cand.get("id") if isinstance(cand, dict) else None
            if pid is None:
                continue
            try:
                prof_resp = await self.tools["get_linkedin_profile"].ainvoke(
                    {"person_id": pid}
                )
                txt = prof_resp[0]["text"] if isinstance(prof_resp, list) else prof_resp
                profile = json.loads(txt) if isinstance(txt, str) else txt
                if not isinstance(profile, dict) or not profile.get("experience"):
                    continue
            except Exception:
                continue
            score = self._match_score(cv, profile)
            if score > best_score:
                best, best_score = profile, score
        return best if best_score > 0 else None

    # ---- Step 4: LLM diff judgement ----------------------------------------

    _DIFF_SYSTEM = (
        "You compare a candidate's CV claims against their LinkedIn profile and "
        "return ONLY a JSON object, no prose.\n"
        'Schema: {"differences": [{"field": str, "cv_value": str, "linkedin_value": str, '
        '"substantive": bool, "severity": "low"|"medium"|"high"}]}\n'
        "Fields to compare: name, city, jobs (company, title, seniority, start/end year), "
        "education (degree, school, field, year), skills.\n"
        "IMPORTANT - wording differences are NOT substantive:\n"
        "- degree: 'BSc' vs 'Bachelor of Science', 'MSc' vs 'Master', 'PhD' vs 'Doctorate' are the same.\n"
        "- title/seniority: 'Engineer' (CV) vs 'Senior Engineer' (profile) with seniority implied is NOT "
        "substantive; but a different rank ('Analyst' vs 'Senior Analyst') when the profile explicitly "
        "marks seniority junior IS substantive.\n"
        "- year: same year, or a rounded range, matches; a 1+ year difference in a start/end year IS "
        "substantive.\n"
        "- skills: a skill on the CV that is absent from the profile IS substantive (medium). "
        "A skill on the profile that the CV simply omits is NOT a difference.\n"
        "- name or city mismatch IS substantive (high).\n"
        'severity: high = wrong person/company/name/city/school; medium = wrong title/rank/degree/field or '
        "missing skill; low = minor wording such as 'UI/UX Design' vs 'UI/UX', a blank vs 'Unknown'.\n"
        "Only list real differences. If CV and profile agree, differences = []."
    )

    async def _judge_diffs(
        self, cv: dict[str, Any], profile: dict[str, Any]
    ) -> list[dict[str, Any]]:
        payload = json.dumps({"cv_claims": cv, "linkedin_profile": profile}, ensure_ascii=False)
        result = await self._llm_json(
            self._DIFF_SYSTEM,
            f"Compare the following. CV claims JSON:\n{payload}",
        )
        diffs = result.get("differences") if isinstance(result, dict) else None
        return diffs if isinstance(diffs, list) else []

    # ---- Step 5: deterministic scoring -------------------------------------

    @staticmethod
    def _score_from_diffs(diffs: list[dict[str, Any]]) -> float:
        substantive = [d for d in diffs if d.get("substantive")]
        if not substantive:
            return 0.95
        penalty = {"low": 0.05, "medium": 0.10, "high": 0.15}
        score = 0.45 - sum(penalty.get(d.get("severity"), 0.10) for d in substantive)
        return max(0.05, round(score, 2))

    # ---- per-CV pipeline ----------------------------------------------------

    async def score_one(self, text: str) -> float:
        cv = await self._extract(text)                      # step 1
        profile = await self._locate(cv)                    # steps 2-3
        if not profile:
            # Could not resolve the person: evidence insufficient -> conservative.
            return 0.3
        diffs = await self._judge_diffs(cv, profile)        # step 4
        return self._score_from_diffs(diffs)                # step 5


async def score_cvs(agent: Any, cvs: dict[str, str]) -> dict[str, float | None]:
    """Run your agent and return one reliability score per CV.

    ``cvs`` maps each file name to its text, e.g. ``{"CV_1.pdf": "...", ...}``.
    Return a float in [0, 1] for every file name: higher means the CV is more
    likely consistent with the candidate's LinkedIn/Facebook data. A score
    above 0.5 counts as "valid", 0.5 or below counts as "has discrepancy".

        {"CV_1.pdf": 0.9, "CV_4.pdf": 0.1, ...}

    Catch errors per CV (e.g. a failed API call) and still return a score for
    it: an exception here means no results.csv, which scores zero.

    MCP tools are async, so call your agent with ``await agent.ainvoke(...)``.
    You may verify CVs in parallel (e.g. ``asyncio.gather``), but keep at most
    about 3 CVs in flight (e.g. with ``asyncio.Semaphore(3)``): the MCP server is
    shared by the whole class.
    """
    semaphore = asyncio.Semaphore(3)

    async def verify_one(name: str, text: str) -> tuple[str, float]:
        async with semaphore:
            try:
                score = await asyncio.wait_for(agent.score_one(text), timeout=90)
                return name, score
            except Exception as exc:
                print(f"[warn] {name} failed ({type(exc).__name__}); conservative 0.3")
                return name, 0.3

    results = await asyncio.gather(*(verify_one(n, t) for n, t in cvs.items()))
    return dict(results)


# Everything below is provided runner/scoring code. No edits are needed.

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_score(value: Any) -> float | None:
    """Accept a float/int, or text containing exactly one number, in [0, 1]."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        text = str(getattr(value, "content", value))
        matches = _NUMBER_RE.findall(text)
        if len(matches) != 1:
            return None
        score = float(matches[0])
    if math.isnan(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read labels (1 = valid CV, 0 = has discrepancy) and reasons from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return {
        name: entry if isinstance(entry, dict) else {"label": entry}
        for name, entry in json.loads(path.read_text(encoding="utf-8")).items()
    }


def correctness_text(score: float | None, expected: dict[str, Any] | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if score is None:
        return "incorrect: score is missing or not a number in [0, 1]"
    if expected is None:
        return "not graded: no ground truth for this CV"
    label = int(expected["label"])
    predicted = 1 if score > THRESHOLD else 0
    if predicted == label:
        return "correct"
    reason = f" ({expected['reason']})" if expected.get("reason") else ""
    return f"incorrect: expected {label}{reason}, predicted {predicted}"


def write_results(names: list[str], scores: dict[str, Any], truth: dict[str, dict[str, Any]]) -> tuple[Path, int]:
    """Write the required results.csv file and return how many CVs were correct."""
    output = Path("results.csv")
    correct = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cv", "score", "correctness"])
        for name in names:
            score = parse_score(scores.get(name))
            verdict = correctness_text(score, truth.get(name))
            correct += verdict == "correct"
            writer.writerow([name, "" if score is None else f"{score:.4f}", verdict])
    return output, correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW2 on CV PDFs")
    parser.add_argument(
        "--cv-folder",
        required=True,
        type=Path,
        help="folder containing CV PDF files",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    paths = cv_files(folder)
    if not paths:
        raise SystemExit(f"no PDF files found in {folder}")

    load_env_file()
    cvs = {path.name: cv_text(path) for path in paths}
    tools = await load_mcp_tools()
    agent = build_agent(tools)
    scores = await score_cvs(agent, cvs)
    if not isinstance(scores, dict):
        raise TypeError("score_cvs() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct = write_results(list(cvs), scores, truth)
    summary = f" Accuracy: {correct}/{len(cvs)}." if truth else ""
    print(f"Processed {len(cvs)} CV(s). Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.cv_folder.is_dir():
        raise SystemExit(f"not a folder: {args.cv_folder}")
    return asyncio.run(run(args.cv_folder))


if __name__ == "__main__":
    raise SystemExit(main())
