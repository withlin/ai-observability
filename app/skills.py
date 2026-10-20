import math
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import yaml
from langchain_core.tools import InjectedToolCallId

from context import current_context
from models import SearchHit, Skill
from telemetry import Telemetry, json_str

DATA_SOURCE_ID = "travel-policy-docs"
SCRIPT_TIMEOUT = 20


@dataclass(frozen=True)
class ScriptResult:
    stdout: str
    stderr: str
    exit_code: int


class SkillRepository:
    """Reads and executes an Agent Skill package from disk."""

    def __init__(self, root: Path):
        self._root = root.resolve()

    def _skill_dir(self, skill_name: str) -> Path:
        if not re.fullmatch(r"[a-z0-9_]+", skill_name or ""):
            raise ValueError(f"invalid skill name: {skill_name!r}")
        path = (self._root / skill_name).resolve()
        if self._root not in path.parents:
            raise ValueError("skill path escapes the skills root")
        if not path.is_dir():
            raise FileNotFoundError(f"skill not found: {skill_name}")
        return path

    @staticmethod
    def _parse_skill_md(path: Path) -> tuple[dict, str]:
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---"):
            raise ValueError("SKILL.md must start with a frontmatter block")
        _, frontmatter, body = text.split("---", 2)
        meta = yaml.safe_load(frontmatter) or {}
        if not isinstance(meta, dict) or not meta.get("name"):
            raise ValueError("SKILL.md frontmatter must contain a name")
        return meta, body.strip()

    def load(self, skill_name: str) -> Skill:
        directory = self._skill_dir(skill_name)
        meta, body = self._parse_skill_md(directory / "SKILL.md")
        return Skill(
            name=str(meta["name"]),
            description=str(meta.get("description", "")),
            instructions=body,
            source_uri=directory.as_uri(),
        )

    def _safe_resource(self, skill_name: str, resource_name: str) -> Path:
        directory = self._skill_dir(skill_name)
        candidate = Path(resource_name)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("resource must be a relative path inside the skill")
        path = (directory / candidate).resolve()
        if directory.resolve() not in path.parents:
            raise ValueError("resource path escapes the skill directory")
        if not path.is_file():
            raise FileNotFoundError(f"resource not found: {resource_name}")
        return path

    def read_resource(self, skill_name: str, resource_name: str) -> str:
        return self._safe_resource(skill_name, resource_name).read_text(encoding="utf-8")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def search(self, skill_name: str, query: str, top_k: int) -> list[SearchHit]:
        directory = self._skill_dir(skill_name)
        docs: list[tuple[str, list[str]]] = []
        for path in sorted((directory / "references").rglob("*.md")):
            rel = path.relative_to(directory).as_posix()
            docs.append((rel, self._tokenize(path.read_text(encoding="utf-8"))))
        query_tokens = self._tokenize(query)
        if not docs or not query_tokens:
            return []

        frequencies = Counter()
        for _, tokens in docs:
            frequencies.update(set(tokens))
        idf = {
            term: math.log((1 + len(docs)) / (1 + count)) + 1
            for term, count in frequencies.items()
        }

        def cosine(tokens: list[str]) -> float:
            counts = Counter(tokens)
            query_counts = Counter(query_tokens)
            numerator = sum(
                query_counts[term] * counts.get(term, 0) * idf.get(term, 1) ** 2
                for term in query_counts
            )
            left = math.sqrt(
                sum((query_counts[term] * idf.get(term, 1)) ** 2 for term in query_counts)
            )
            right = math.sqrt(
                sum((counts[term] * idf.get(term, 1)) ** 2 for term in counts)
            )
            return numerator / (left * right) if left and right else 0.0

        ranked = sorted(
            ((cosine(tokens), rel) for rel, tokens in docs),
            key=lambda item: item[0],
            reverse=True,
        )
        hits = []
        for score, rel in ranked[:max(1, top_k)]:
            text = (directory / rel).read_text(encoding="utf-8")
            hits.append(SearchHit(id=rel, score=round(score, 4), snippet=text[:500]))
        return hits

    def run_script(self, skill_name: str, script_name: str, cli_args: list[str]) -> ScriptResult:
        path = self._safe_resource(skill_name, script_name)
        directory = self._skill_dir(skill_name)
        if path.suffix != ".py" or "scripts" not in path.relative_to(directory).parts:
            raise ValueError("only scripts/*.py may be executed")
        completed = subprocess.run(
            [sys.executable, str(path), *cli_args],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=SCRIPT_TIMEOUT,
            shell=False,
        )
        return ScriptResult(
            stdout=completed.stdout.strip(),
            stderr=completed.stderr.strip(),
            exit_code=completed.returncode,
        )


class SkillTools:
    """LangChain tools that expose one Agent Skill to a specific agent."""

    def __init__(self, repository: SkillRepository, telemetry: Telemetry,
                 skill_name: str = "travel_policy", agent_name: str = "policy_agent"):
        self._repository = repository
        self._telemetry = telemetry
        self._skill_name = skill_name
        self._agent_name = agent_name

    def tools(self) -> list:
        return [self.load_skill, self.search_skill_resources,
                self.read_skill_resource, self.estimate_budget]

    def load_skill(self, skill_name: str = "travel_policy",
                   tool_call_id: Annotated[str, InjectedToolCallId] = "") -> str:
        """Load an Agent Skill into the agent context and return its instructions."""
        skill = self._repository.load(skill_name)
        context = current_context()
        with self._telemetry.tool_span(
            "load_skill",
            f"execute_tool load_skill {skill.name}",
            tool_call_id,
            {"skill_name": skill_name},
            description=self.load_skill.__doc__,
            agent_name=self._agent_name,
            context=context,
            attrs={
                "gen_ai.skill.name": skill.name,
                "gen_ai.skill.description": skill.description,
                "gen_ai.skill.source.uri": skill.source_uri,
            },
        ) as span:
            result = f"# {skill.name}\n\n{skill.instructions}"
            span.set_result(result)
            return result

    def search_skill_resources(self, skill_name: str = "travel_policy", query: str = "",
                               top_k: int = 5,
                               tool_call_id: Annotated[str, InjectedToolCallId] = "") -> str:
        """Search the skill's reference documents and return the top matching snippets."""
        context = current_context()
        with self._telemetry.tool_span(
            "search_skill_resources",
            "execute_tool search_skill_resources",
            tool_call_id,
            {"skill_name": skill_name, "query": query, "top_k": top_k},
            description=self.search_skill_resources.__doc__,
            agent_name=self._agent_name,
            context=context,
            attrs={"gen_ai.skill.name": skill_name},
        ) as tool_span:
            with self._telemetry.retrieval_span(
                data_source_id=DATA_SOURCE_ID,
                top_k=top_k,
                query=query,
                agent_name=self._agent_name,
                context=context,
            ) as retrieval_span:
                hits = self._repository.search(skill_name, query, top_k)
                if self._telemetry.capture_content:
                    retrieval_span.set_attribute(
                        "gen_ai.retrieval.documents",
                        json_str([{"id": hit.id, "score": hit.score} for hit in hits]),
                    )
            payload = json_str({
                "documents": [
                    {"id": hit.id, "score": hit.score, "snippet": hit.snippet}
                    for hit in hits
                ]
            })
            tool_span.set_result(payload)
            return payload

    def read_skill_resource(self, skill_name: str = "travel_policy",
                            resource_name: str = "",
                            tool_call_id: Annotated[str, InjectedToolCallId] = "") -> str:
        """Read a reference, asset, or script file bundled with an Agent Skill."""
        skill = self._repository.load(skill_name)
        context = current_context()
        with self._telemetry.tool_span(
            "read_skill_resource",
            f"execute_tool read_skill_resource {skill.name} {resource_name}",
            tool_call_id,
            {"skill_name": skill_name, "resource_name": resource_name},
            description=self.read_skill_resource.__doc__,
            agent_name=self._agent_name,
            context=context,
            attrs={
                "gen_ai.skill.name": skill.name,
                "gen_ai.skill.resource.name": resource_name,
                "gen_ai.skill.description": skill.description,
                "gen_ai.skill.source.uri": skill.source_uri,
            },
        ) as span:
            result = self._repository.read_resource(skill_name, resource_name)
            span.set_result(result)
            return result

    def estimate_budget(self, city: str, days: int, style: str = "mid-range",
                        tool_call_id: Annotated[str, InjectedToolCallId] = "") -> str:
        """Estimate a daily and total trip budget by running the travel_policy budget script."""
        return self._run_script(
            skill_name=self._skill_name,
            script_name="scripts/estimate_budget.py",
            cli_args=["--city", city, "--days", str(days), "--style", style],
            tool_name="estimate_budget",
            call_args={"city": city, "days": days, "style": style},
            description=self.estimate_budget.__doc__,
            tool_call_id=tool_call_id,
        )

    def run_skill_script(self, skill_name: str, script_name: str, args: list[str],
                         tool_call_id: Annotated[str, InjectedToolCallId] = "") -> str:
        """Execute a skill script. For budgets prefer estimate_budget."""
        return self._run_script(
            skill_name=skill_name,
            script_name=script_name,
            cli_args=args,
            tool_name="run_skill_script",
            call_args={"skill_name": skill_name, "script_name": script_name, "args": args},
            description=self.run_skill_script.__doc__,
            tool_call_id=tool_call_id,
        )

    def _run_script(self, *, skill_name: str, script_name: str, cli_args: list[str],
                    tool_name: str, call_args: dict, description: str,
                    tool_call_id: str) -> str:
        skill = self._repository.load(skill_name)
        context = current_context()
        with self._telemetry.tool_span(
            tool_name,
            f"execute_tool {tool_name} {skill.name} {script_name}",
            tool_call_id,
            call_args,
            description=description,
            agent_name=self._agent_name,
            context=context,
            attrs={
                "gen_ai.skill.name": skill.name,
                "gen_ai.skill.resource.name": script_name,
                "gen_ai.skill.description": skill.description,
                "gen_ai.skill.source.uri": skill.source_uri,
                "process.executable.name": Path(sys.executable).name,
                "process.executable.path": sys.executable,
            },
        ) as span:
            result = self._repository.run_script(skill_name, script_name, cli_args)
            span.set_attribute("process.exit.code", result.exit_code)
            if result.exit_code != 0:
                raise RuntimeError(result.stderr or f"script exited {result.exit_code}")
            span.set_result(result.stdout)
            return result.stdout
