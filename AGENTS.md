# CARE:HERO local analysis guide

- Keep landmark extraction, rule scoring, and LLM wording in separate modules.
- Raw measurements and rule scores are authoritative; the LLM must never modify them.
- Store every rule threshold in scenario YAML and label unvalidated values provisional.
- If detection data is insufficient, return no total score instead of a normal value.
- Send compact rule evidence to the LLM, never the full frame/keypoint sequence.
- Require structured LLM output validated by Pydantic.
- Do not make medical, diagnostic, employment, or care-suitability claims.
- Do not log API keys, video content, keypoint dumps, or personal file paths.
- Use Python 3.11+, type hints, pytest, and deterministic tests with mocked LLM calls.

