---
name: travel_policy
description: Entry requirements, baggage rules, and daily budget guidance for common travel destinations.
---

# Travel Policy

Use this skill when the traveler asks about visas, entry requirements, baggage
allowances, or a reasonable daily budget.

## Procedure

1. Search the references for the destination and topic:
   `search_skill_resources(skill_name="travel_policy", query="Japan visa entry requirements")`.
2. Read the most relevant reference in full:
   `read_skill_resource(skill_name="travel_policy", resource_name="references/entry_requirements.md")`.
3. When the traveler asks for a budget, call:
   `estimate_budget(city="Tokyo", days=5, style="mid-range")`.
4. Answer only from the skill content and tool results. If a fact is missing,
   say that it could not be verified.
