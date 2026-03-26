import argparse
import json
import os
import re
import textwrap
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from html import unescape
from typing import Any, Iterable
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = 15
SEARCH_DELAY_SECONDS = 1.0
MAX_CANDIDATES = 24
MAX_REPORT_MATCHES = 8
SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"
SEARCH_ENDPOINT_BING = "https://www.bing.com/search"
DEFAULT_MARKDOWN_OUT = "lead_report.md"
DEFAULT_JSON_OUT = "lead_report.json"
MIN_REPORT_SCORE = 35
MIN_OUTREACH_SCORE = 55
MIN_ACCEPTED_PROFILE_SCORE = 55
DEFAULT_EXECUTION_MODE = "auto"
DEFAULT_AGENT_MODEL = "llama-3.1-8b-instant"
LOW_MEMORY_AGENT_MODEL = "llama-3.1-8b-instant"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODELS_URL = f"{GROQ_BASE_URL}/models"
GROQ_CHAT_URL = f"{GROQ_BASE_URL}/chat/completions"
GROQ_TIMEOUT = 45
AGENT_MAX_SEARCH_ROUNDS = 2
MAX_PLANNED_QUERIES = 8
STOP_AFTER_ACCEPTED_MATCHES = 2
MIN_AGENT_HEURISTIC_SCORE = MIN_REPORT_SCORE
MAX_DIRECT_PROBES = 24

SOCIAL_DOMAINS = {
    "github.com": "GitHub",
    "x.com": "X",
    "twitter.com": "Twitter",
    "medium.com": "Medium",
    "dev.to": "Dev.to",
    "substack.com": "Substack",
    "youtube.com": "YouTube",
    "about.me": "About.me",
    "instagram.com": "Instagram",
    "facebook.com": "Facebook",
    "crunchbase.com": "Crunchbase",
    "angel.co": "AngelList",
    "wellfound.com": "Wellfound",
    "wikipedia.org": "Wikipedia",
    "britannica.com": "Britannica",
}

PROFILE_RESERVED_SEGMENTS = {
    "github.com": {"about", "collections", "events", "features", "login", "marketplace", "notifications", "orgs", "search", "settings", "topics"},
    "x.com": {"compose", "explore", "hashtag", "home", "i", "intent", "messages", "search", "settings", "share"},
    "twitter.com": {"compose", "explore", "hashtag", "home", "i", "intent", "messages", "search", "settings", "share"},
    "instagram.com": {"about", "accounts", "developer", "directory", "explore", "p", "reel", "stories"},
    "facebook.com": {"events", "gaming", "groups", "help", "marketplace", "pages", "search", "watch"},
    "medium.com": {"about", "m", "search", "tag", "topics"},
    "dev.to": {"about", "latest", "search", "tags", "top"},
    "youtube.com": {"feed", "playlist", "results", "shorts", "watch"},
    "about.me": set(),
    "wellfound.com": {"jobs", "recruit", "discover"},
    "angel.co": {"company", "jobs"},
    "crunchbase.com": {"organization", "discover"},
}

SEARCH_LIKE_SEGMENTS = {
    "directory",
    "discover",
    "examples",
    "explore",
    "find",
    "lookup",
    "query",
    "results",
    "s",
    "search",
    "tag",
    "topics",
}

GENERIC_EVIDENCE_TERMS = {
    "english examples",
    "example sentences",
    "results for",
    "search results",
    "translation",
}

PROFILE_META_TERMS = {
    "about",
    "bio",
    "engineer",
    "experience",
    "founder",
    "portfolio",
    "profile",
    "software",
}

PERSON_SCHEMA_TYPES = {"person", "profilepage"}

ROLE_STOPWORDS = {
    "and",
    "at",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
    "a",
    "an",
}


@dataclass
class LeadProfile:
    linkedin_url: str
    full_name: str = ""
    first_name: str = ""
    last_name: str = ""
    headline: str = ""
    company: str = ""
    location: str = ""
    slug: str = ""
    search_queries: list[str] = field(default_factory=list)
    source_notes: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source_query: str
    search_provider: str = ""
    discovery_method: str = "search"


@dataclass
class Evidence:
    url: str
    title: str
    description: str
    text_excerpt: str
    domain: str
    platform: str
    page_type: str = "unknown"
    is_profile_like: bool = False
    is_directory_like: bool = False
    profile_signals: list[str] = field(default_factory=list)
    warning_signals: list[str] = field(default_factory=list)
    outgoing_links: list[str] = field(default_factory=list)


@dataclass
class MatchResult:
    candidate_url: str
    platform: str
    confidence_score: int
    confidence_label: str
    reasons: list[str]
    evidence: Evidence


@dataclass
class CandidateFacts:
    display_name: str = ""
    handle: str = ""
    bio: str = ""
    company: str = ""
    location: str = ""
    links: list[str] = field(default_factory=list)
    evidence_snippets: list[str] = field(default_factory=list)
    extraction_notes: list[str] = field(default_factory=list)
    normalized_by: str = "heuristic"


@dataclass
class CandidateDecision:
    status: str
    rationale: str
    matched_signals: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    review_notes: list[str] = field(default_factory=list)
    decided_by: str = "heuristic"


@dataclass
class CandidateAssessment:
    match: MatchResult
    facts: CandidateFacts
    decision: CandidateDecision


@dataclass
class ExecutionContext:
    requested_mode: str
    actual_mode: str
    requested_model: str
    active_model: str = ""
    groq_available: bool = False
    fallback_reason: str = ""
    provider_name: str = "groq"
    trace: list[dict[str, Any]] = field(default_factory=list)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (text or "").lower()).strip()


def sanitize_person_name(text: str) -> str:
    cleaned = strip_broken_encoding(text)
    cleaned = re.sub(r"[^\w\s'.&/-]+", "", cleaned, flags=re.UNICODE)
    cleaned = cleaned.replace("_", " ")
    return normalize_whitespace(cleaned)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 1]


def strip_broken_encoding(text: str) -> str:
    return normalize_whitespace((text or "").replace("Â·", " ").replace("â€“", "-").replace("â€™", "'"))


def url_path_segments(url: str) -> list[str]:
    return [segment for segment in urlparse(url).path.strip("/").split("/") if segment]


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen = set()
    items = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            items.append(value)
    return items


def add_trace(trace: list[dict[str, Any]], step: str, summary: str, data: dict[str, Any] | None = None) -> None:
    trace.append(
        {
            "step": step,
            "summary": summary,
            "data": data or {},
        }
    )


def safe_json_loads(raw_text: str) -> dict[str, Any] | list[Any] | None:
    if not raw_text:
        return None

    text = raw_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            return None

    start_object = text.find("{")
    end_object = text.rfind("}")
    if start_object != -1 and end_object != -1 and end_object > start_object:
        try:
            return json.loads(text[start_object : end_object + 1])
        except json.JSONDecodeError:
            return None

    start_array = text.find("[")
    end_array = text.rfind("]")
    if start_array != -1 and end_array != -1 and end_array > start_array:
        try:
            return json.loads(text[start_array : end_array + 1])
        except json.JSONDecodeError:
            return None

    return None


def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
    return session


def safe_get(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def get_groq_api_key() -> str:
    return normalize_whitespace(os.environ.get("GROQ_API_KEY", ""))


def get_groq_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def get_groq_models(session: requests.Session, api_key: str) -> list[str]:
    if not api_key:
        return []
    try:
        response = session.get(GROQ_MODELS_URL, headers=get_groq_headers(api_key), timeout=10)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return []

    models = payload.get("data", [])
    names = []
    for model in models:
        if isinstance(model, dict) and model.get("id"):
            names.append(str(model["id"]))
    return names


def resolve_execution_context(
    session: requests.Session, requested_mode: str, requested_model: str | None
) -> ExecutionContext:
    api_key = get_groq_api_key()
    available_models = get_groq_models(session, api_key)
    groq_available = bool(available_models)
    requested_model_name = (requested_model or DEFAULT_AGENT_MODEL).strip()
    context = ExecutionContext(
        requested_mode=requested_mode,
        actual_mode="fallback",
        requested_model=requested_model_name,
        groq_available=groq_available,
    )

    if requested_mode == "fallback":
        context.actual_mode = "fallback"
        context.fallback_reason = "Fallback mode was explicitly requested."
        return context

    if not api_key:
        context.actual_mode = "fallback"
        context.fallback_reason = "GROQ_API_KEY is not set, so the deterministic fallback pipeline was used."
        return context

    if not groq_available:
        context.actual_mode = "fallback"
        context.fallback_reason = "Groq API is unreachable or the API key is invalid, so the deterministic fallback pipeline was used."
        return context

    preferred_models = [requested_model_name]
    if not requested_model and LOW_MEMORY_AGENT_MODEL not in preferred_models:
        preferred_models.append(LOW_MEMORY_AGENT_MODEL)

    installed_lookup = {name.lower(): name for name in available_models}
    for preferred_model in preferred_models:
        if preferred_model.lower() in installed_lookup:
            context.active_model = installed_lookup[preferred_model.lower()]
            context.actual_mode = "agent"
            return context

    if requested_mode == "agent":
        context.actual_mode = "fallback"
        context.fallback_reason = (
            f"Groq is reachable but the requested model '{requested_model_name}' is not available for this API key."
        )
        return context

    if LOW_MEMORY_AGENT_MODEL.lower() in installed_lookup:
        context.active_model = installed_lookup[LOW_MEMORY_AGENT_MODEL.lower()]
        context.actual_mode = "agent"
        return context

    context.actual_mode = "fallback"
    context.fallback_reason = (
        f"Groq is reachable but none of the preferred models are available ({', '.join(preferred_models)})."
    )
    return context


def groq_chat_json(
    session: requests.Session,
    model: str,
    system_prompt: str,
    user_prompt: str,
    trace: list[dict[str, Any]],
    step_name: str,
) -> dict[str, Any] | None:
    api_key = get_groq_api_key()
    if not api_key:
        add_trace(trace, step_name, "Groq request skipped because GROQ_API_KEY is missing.", {"model": model})
        return None
    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    try:
        response = session.post(GROQ_CHAT_URL, headers=get_groq_headers(api_key), json=payload, timeout=GROQ_TIMEOUT)
        response.raise_for_status()
        content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    except (requests.RequestException, ValueError) as exc:
        add_trace(trace, step_name, "Groq request failed.", {"error": str(exc), "model": model})
        return None

    parsed = safe_json_loads(content)
    if isinstance(parsed, dict):
        add_trace(trace, step_name, "Groq returned structured JSON.", {"model": model})
        return parsed

    add_trace(trace, step_name, "Groq response was not valid JSON.", {"model": model, "raw": content[:500]})
    return None


def parse_linkedin_slug(linkedin_url: str) -> str:
    path = urlparse(linkedin_url).path.strip("/")
    parts = [p for p in path.split("/") if p]
    if not parts:
        return ""
    return parts[-1]


def slug_name_tokens(slug: str) -> list[str]:
    tokens = []
    for token in slug.replace("-", " ").split():
        cleaned = re.sub(r"[^a-zA-Z]", "", token)
        if len(cleaned) >= 2:
            tokens.append(cleaned)
    return tokens


def parse_headline_company(headline: str) -> tuple[str, str]:
    headline = normalize_whitespace(headline)
    if not headline:
        return "", ""

    separators = [" @ ", " at ", " | ", " - "]
    for separator in separators:
        if separator in headline:
            left, right = headline.split(separator, 1)
            role = normalize_whitespace(left)
            company = normalize_whitespace(right)
            if role and company:
                return role, company
    return headline, ""


def extract_linkedin_meta_fields(description: str) -> tuple[str, str, str]:
    cleaned = unescape(description or "")
    cleaned = re.sub(r"View .* profile on LinkedIn\.?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("\r", "")
    cleaned = cleaned.replace("Ã‚Â·", " ").replace("Ã¢â‚¬â€œ", "-").replace("Ã¢â‚¬â„¢", "'")

    company = ""
    location = ""
    headline = ""

    company_match = re.search(
        r"(?:Experience|Current company|Company):\s*([^·\n|]{1,80})",
        cleaned,
        flags=re.IGNORECASE,
    )
    if company_match:
        company = sanitize_company_text(company_match.group(1))

    location_match = re.search(r"Location:\s*([^·\n|]{1,60})", cleaned, flags=re.IGNORECASE)
    if location_match:
        location = sanitize_location_text(location_match.group(1))

    preface = cleaned.split("· Experience:", 1)[0]
    paragraphs = [normalize_whitespace(part) for part in re.split(r"\n{2,}", preface) if normalize_whitespace(part)]
    if paragraphs:
        headline = sanitize_headline_text(paragraphs[0].split(". ", 1)[0])
    elif cleaned:
        headline = sanitize_headline_text(cleaned.split("·", 1)[0])

    return headline, company, location


def sanitize_company_text(company: str) -> str:
    company = strip_broken_encoding(company)
    if not company:
        return ""
    handles = re.findall(r"@[A-Za-z0-9][A-Za-z0-9._&\- ]{1,40}", company)
    if handles:
        cleaned_handles = []
        for handle in handles[:2]:
            normalized = normalize_whitespace(handle.replace("@", "").replace("  ", " "))
            if normalized and len(normalized) <= 30:
                cleaned_handles.append(normalized)
        return " / ".join(cleaned_handles)

    if len(company) > 60:
        return ""
    return company


def sanitize_location_text(location: str) -> str:
    location = strip_broken_encoding(location)
    if not location:
        return ""

    match = re.search(r"Location:\s*([A-Za-z][A-Za-z ,.\-]{1,60})", location, flags=re.IGNORECASE)
    if match:
        candidate = normalize_whitespace(match.group(1))
        if len(candidate) <= 40:
            return candidate

    lowered = location.lower()
    if any(token in lowered for token in ["experience:", "education:", "connections on linkedin"]):
        return ""

    if len(location) > 50:
        return ""
    return location


def sanitize_headline_text(headline: str) -> str:
    headline = strip_broken_encoding(headline)
    if not headline:
        return ""
    return normalize_whitespace(headline[:120])


def parse_linkedin_structured_person(soup: BeautifulSoup) -> dict[str, str]:
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.get_text(strip=True)
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if isinstance(payload, dict) and isinstance(payload.get("@graph"), list):
            nodes = payload["@graph"]
        elif isinstance(payload, list):
            nodes = payload
        else:
            nodes = [payload]

        for node in nodes:
            if not isinstance(node, dict) or node.get("@type") != "Person":
                continue

            companies = []
            works_for = node.get("worksFor") or []
            if isinstance(works_for, dict):
                works_for = [works_for]
            for item in works_for:
                if not isinstance(item, dict):
                    continue
                name = sanitize_company_text(str(item.get("name", "")))
                if name and "*" not in name:
                    companies.append(name)

            address = node.get("address") or {}
            location = ""
            if isinstance(address, dict):
                locality = normalize_whitespace(str(address.get("addressLocality", "")))
                country = normalize_whitespace(str(address.get("addressCountry", "")))
                if locality and country and len(country) <= 3:
                    location_seed = locality
                else:
                    location_seed = " ".join(part for part in [locality, country] if part)
                location = sanitize_location_text(
                    location_seed
                )

            return {
                "name": sanitize_person_name(str(node.get("name", ""))),
                "company": companies[0] if companies else "",
                "location": location,
            }

    return {}


def profile_name_tokens(profile: LeadProfile) -> list[str]:
    name = profile.full_name or " ".join(part for part in [profile.first_name, profile.last_name] if part)
    return [token for token in tokenize(name) if token not in {"mr", "mrs", "ms", "dr"}]


def social_search_domains() -> list[str]:
    return [
        "x.com",
        "twitter.com",
        "instagram.com",
        "facebook.com",
        "youtube.com",
        "medium.com",
        "substack.com",
        "about.me",
        "github.com",
        "dev.to",
    ]


def website_search_queries(profile: LeadProfile) -> list[str]:
    name = profile.full_name or " ".join(part for part in [profile.first_name, profile.last_name] if part)
    if not name:
        return []

    queries = [
        f'"{name}" ("official site" OR website OR blog OR newsletter OR portfolio)',
        f'"{name}" (site:medium.com OR site:substack.com OR site:about.me OR site:github.com OR site:dev.to)',
        f'"{name}" ("author" OR "team" OR "speaker" OR "bio")',
    ]
    if profile.company:
        queries.append(f'"{name}" "{profile.company}" ("team" OR "bio" OR "author")')
    return queries


def direct_probe_patterns() -> list[tuple[str, str]]:
    return [
        ("X", "https://x.com/{handle}"),
        ("Twitter", "https://twitter.com/{handle}"),
        ("Instagram", "https://www.instagram.com/{handle}/"),
        ("Facebook", "https://www.facebook.com/{handle}/"),
        ("GitHub", "https://github.com/{handle}"),
        ("Dev.to", "https://dev.to/{handle}"),
        ("Medium", "https://medium.com/@{handle}"),
        ("About.me", "https://about.me/{handle}"),
        ("YouTube", "https://www.youtube.com/@{handle}"),
        ("Substack", "https://{handle}.substack.com"),
    ]


def build_handle_candidates(profile: LeadProfile, discovered_clues: dict[str, str]) -> list[str]:
    tokens = profile_name_tokens(profile)
    candidates: list[str] = []
    if not tokens:
        return candidates

    compact = "".join(tokens)
    if compact:
        candidates.extend([compact, "_".join(tokens), ".".join(tokens), "-".join(tokens)])

    if len(tokens) >= 2:
        candidates.extend(
            [
                tokens[0] + tokens[-1],
                tokens[0] + "_" + tokens[-1],
                tokens[0] + "." + tokens[-1],
                tokens[0] + "-" + tokens[-1],
                tokens[0] + tokens[1][0] + tokens[-1] if len(tokens) >= 3 else "",
            ]
        )

    slug_candidates = slug_name_tokens(profile.slug)
    if slug_candidates:
        slug_compact = "".join(tokenize(" ".join(slug_candidates)))
        if slug_compact:
            candidates.append(slug_compact)

    discovered_handle = normalize_whitespace(discovered_clues.get("handle", ""))
    if discovered_handle:
        candidates.append(discovered_handle.replace("@", ""))

    cleaned = []
    for candidate in candidates:
        normalized = re.sub(r"[^a-zA-Z0-9._-]", "", candidate or "").lower()
        if 3 <= len(normalized) <= 30:
            cleaned.append(normalized)
    return unique_preserve_order(cleaned)


def extract_linkedin_profile(session: requests.Session, linkedin_url: str) -> LeadProfile:
    profile = LeadProfile(linkedin_url=linkedin_url, slug=parse_linkedin_slug(linkedin_url))
    profile.source_notes.append("Seeded from the LinkedIn URL slug.")

    slug_tokens = slug_name_tokens(profile.slug)
    if slug_tokens:
        profile.full_name = " ".join(token.capitalize() for token in slug_tokens[:4])
        if len(slug_tokens) >= 2:
            profile.first_name = slug_tokens[0].capitalize()
            profile.last_name = slug_tokens[-1].capitalize()

    try:
        html = safe_get(session, linkedin_url)
    except requests.RequestException as exc:
        profile.source_notes.append(f"Could not fetch LinkedIn page directly: {exc}")
        return profile

    soup = BeautifulSoup(html, "html.parser")
    title = normalize_whitespace(soup.title.get_text(" ", strip=True) if soup.title else "")
    structured_person = parse_linkedin_structured_person(soup)

    og_title = ""
    og_description = ""
    title_tag = soup.find("meta", attrs={"property": "og:title"})
    description_tag = soup.find("meta", attrs={"name": "description"}) or soup.find(
        "meta", attrs={"property": "og:description"}
    )
    if title_tag and title_tag.get("content"):
        og_title = normalize_whitespace(title_tag["content"])
    if description_tag and description_tag.get("content"):
        og_description = normalize_whitespace(description_tag["content"])

    raw_title = og_title or title
    raw_description = og_description

    if structured_person.get("name"):
        profile.full_name = structured_person["name"]
        parts = profile.full_name.split()
        if parts:
            profile.first_name = parts[0]
            profile.last_name = parts[-1]
        profile.source_notes.append("Parsed name from LinkedIn structured data.")
    if structured_person.get("company"):
        profile.company = structured_person["company"]
        profile.source_notes.append("Parsed company from LinkedIn structured data.")
    if structured_person.get("location"):
        profile.location = structured_person["location"]
        profile.source_notes.append("Parsed location from LinkedIn structured data.")

    if raw_title:
        name_candidate = raw_title.split("|")[0].split("-")[0].strip()
        if len(name_candidate.split()) >= 2:
            profile.full_name = sanitize_person_name(name_candidate)
            parts = profile.full_name.split()
            profile.first_name = parts[0]
            profile.last_name = parts[-1]
            profile.source_notes.append("Parsed name from the LinkedIn page title.")

    if raw_description:
        headline, company, location = extract_linkedin_meta_fields(raw_description)
        if headline:
            role, inline_company = parse_headline_company(headline)
            profile.headline = sanitize_headline_text(role or headline)
            if inline_company and not profile.company:
                profile.company = sanitize_company_text(inline_company)
        if company and not profile.company:
            profile.company = sanitize_company_text(company)
        if location and not profile.location:
            profile.location = sanitize_location_text(location)
        profile.source_notes.append("Parsed headline/company/location from LinkedIn meta description.")

    if raw_title:
        title_without_suffix = normalize_whitespace(raw_title.split("|", 1)[0])
        title_parts = [normalize_whitespace(p) for p in title_without_suffix.split("-") if normalize_whitespace(p)]
        if len(title_parts) >= 2 and not profile.company:
            profile.company = sanitize_company_text(title_parts[1])

    profile.headline = sanitize_headline_text(profile.headline)
    profile.company = sanitize_company_text(profile.company)
    profile.location = sanitize_location_text(profile.location)
    profile.full_name = sanitize_person_name(profile.full_name)

    if not any([profile.headline, profile.company, profile.location]):
        profile.source_notes.append(
            "LinkedIn did not expose enough public metadata, so enrichment relied mostly on slug-based search."
        )

    return profile


def build_search_queries(profile: LeadProfile) -> list[str]:
    queries = []
    name = profile.full_name or " ".join(
        part for part in [profile.first_name, profile.last_name] if part
    )
    social_domains = social_search_domains()

    if name:
        for domain in social_domains[:8]:
            queries.append(f'"{name}" site:{domain}')
        queries.extend(website_search_queries(profile))
        queries.append(f'"{name}" official')
        if profile.company:
            queries.append(f'"{name}" "{profile.company}"')
        if profile.location:
            queries.append(f'"{name}" "{profile.location}"')

    if profile.slug:
        slug_as_name = " ".join(slug_name_tokens(profile.slug)) or profile.slug.replace("-", " ")
        queries.append(f'"{slug_as_name}" site:x.com')
        queries.append(f'"{slug_as_name}" site:twitter.com')

    profile.search_queries = unique_preserve_order(query.strip() for query in queries if query.strip())
    return profile.search_queries


def build_refined_queries(profile: LeadProfile, discovered_clues: dict[str, str]) -> list[str]:
    queries = []
    name = profile.full_name or " ".join(part for part in [profile.first_name, profile.last_name] if part)

    if not name:
        return []

    handle = discovered_clues.get("handle", "").strip()
    company = discovered_clues.get("company", "").strip() or profile.company
    role = discovered_clues.get("role", "").strip() or profile.headline
    location = discovered_clues.get("location", "").strip() or profile.location
    website = discovered_clues.get("website", "").strip()

    if handle:
        queries.append(f'"{handle}" site:x.com')
        queries.append(f'"{handle}" site:twitter.com')
        queries.append(f'"{handle}" site:instagram.com')
        queries.append(f'"{handle}" site:github.com')
        queries.append(f'"{handle}" site:medium.com')
    if company:
        queries.append(f'"{name}" "{company}" site:x.com')
        queries.append(f'"{name}" "{company}" site:youtube.com')
        queries.append(f'"{name}" "{company}" ("team" OR "bio" OR "author")')
    if role:
        queries.append(f'"{name}" "{role}"')
    if location:
        queries.append(f'"{name}" "{location}"')
    if website:
        domain = urlparse(website).netloc.lower().replace("www.", "")
        if domain:
            queries.append(f'"{name}" site:{domain}')
    queries.append(f'"{name}" ("blog" OR "newsletter" OR "official site")')

    return unique_preserve_order(query.strip() for query in queries if query.strip())


def plan_search_queries(
    profile: LeadProfile,
    context: ExecutionContext,
    session: requests.Session,
    round_index: int,
    discovered_clues: dict[str, str],
) -> list[str]:
    if context.actual_mode != "agent" or not context.active_model:
        if round_index == 0:
            queries = build_search_queries(profile)[:MAX_PLANNED_QUERIES]
        else:
            queries = build_refined_queries(profile, discovered_clues)[:MAX_PLANNED_QUERIES]
        add_trace(context.trace, "search_planner", "Used heuristic search planner.", {"round": round_index + 1, "queries": queries})
        return queries

    default_queries = build_search_queries(profile)[:MAX_PLANNED_QUERIES]
    query_candidates = default_queries if round_index == 0 else build_refined_queries(profile, discovered_clues)[:MAX_PLANNED_QUERIES]
    prompt = textwrap.dedent(
        f"""
        Select up to {MAX_PLANNED_QUERIES} search queries for finding this person's public social profiles and personal websites.
        Only choose from the candidate queries provided below.
        Prefer exact-name queries targeting X/Twitter, Instagram, Facebook, YouTube, Medium, Substack, About.me, and official pages.
        Reject noisy queries containing long LinkedIn fragments, malformed prefixes, fake domains, or huge quoted blobs.

        Lead seed:
        - name: {profile.full_name or 'unknown'}
        - headline: {profile.headline or 'unknown'}
        - company: {profile.company or 'unknown'}
        - location: {profile.location or 'unknown'}
        - linkedin slug: {profile.slug or 'unknown'}
        - search round: {round_index + 1}
        - discovered clues: {json.dumps(discovered_clues, ensure_ascii=False)}
        - candidate queries: {json.dumps(query_candidates, ensure_ascii=False)}

        Return JSON only:
        {{
          "queries": ["..."],
          "reasoning": "short reason"
        }}
        """
    ).strip()
    system_prompt = "You are a search-planning agent. Return only strict JSON."
    parsed = groq_chat_json(session, context.active_model, system_prompt, prompt, context.trace, "search_planner")
    raw_queries = parsed.get("queries", []) if isinstance(parsed, dict) else []
    queries = [normalize_whitespace(str(item)) for item in raw_queries if normalize_whitespace(str(item))]
    allowed = set(query_candidates)
    sanitized = []
    for query in queries:
        if query not in allowed:
            continue
        if "+" in query[:2]:
            continue
        if "site:personalwebsites" in query:
            continue
        if len(query) > 160:
            continue
        sanitized.append(query)
    queries = unique_preserve_order(sanitized)[:MAX_PLANNED_QUERIES]

    if not queries:
        queries = query_candidates

    add_trace(
        context.trace,
        "search_planner",
        "Planned search queries.",
        {"round": round_index + 1, "queries": queries, "discovered_clues": discovered_clues},
    )
    return queries


def unwrap_duckduckgo_url(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return unquote(query["uddg"][0])
    return url


def is_search_blocked(provider: str, status_code: int, html_text: str) -> bool:
    lowered = (html_text or "").lower()
    if provider == "duckduckgo":
        if status_code == 202 and any(token in lowered for token in ["anomaly", "challenge", "unfortunately"]):
            return True
    if provider == "bing":
        if "captcha" in lowered and "b_algo" not in lowered:
            return True
    return False


def is_valid_linkedin_profile_url(url: str) -> bool:
    parsed = urlparse(url.strip())
    domain = parsed.netloc.lower().replace("www.", "")
    path = parsed.path.strip("/")
    return domain == "linkedin.com" and path.startswith("in/") and len(path.split("/")) >= 2


def prompt_for_linkedin_url() -> str:
    while True:
        candidate = input("Enter LinkedIn profile URL: ").strip()
        if not candidate:
            print("A LinkedIn profile URL is required.")
            continue
        if not is_valid_linkedin_profile_url(candidate):
            print("Please enter a valid LinkedIn profile URL in the format https://www.linkedin.com/in/...")
            continue
        return candidate


def prompt_for_output_path(label: str, default_path: str) -> str:
    entered = input(f"Enter {label} [{default_path}]: ").strip()
    return entered or default_path


def resolve_cli_inputs(args: argparse.Namespace) -> tuple[str, str, str]:
    interactive_mode = not args.linkedin_url

    if interactive_mode:
        linkedin_url = prompt_for_linkedin_url()
        markdown_out = args.markdown_out or prompt_for_output_path("Markdown output filename", DEFAULT_MARKDOWN_OUT)
        json_out = args.json_out or prompt_for_output_path("JSON output filename", DEFAULT_JSON_OUT)
    else:
        linkedin_url = args.linkedin_url.strip()
        if not is_valid_linkedin_profile_url(linkedin_url):
            raise SystemExit("Please provide a valid LinkedIn profile URL in the format https://www.linkedin.com/in/...")
        markdown_out = args.markdown_out or DEFAULT_MARKDOWN_OUT
        json_out = args.json_out or DEFAULT_JSON_OUT

    return linkedin_url, markdown_out, json_out


def search_duckduckgo(session: requests.Session, query: str) -> tuple[str, list[SearchResult]]:
    response = session.post(
        SEARCH_ENDPOINT,
        data={"q": query},
        timeout=REQUEST_TIMEOUT,
        headers={"Referer": "https://duckduckgo.com/"},
    )
    response.raise_for_status()

    if is_search_blocked("duckduckgo", response.status_code, response.text):
        return "blocked", []

    soup = BeautifulSoup(response.text, "html.parser")
    results = []

    for card in soup.select(".result"):
        link = card.select_one(".result__a")
        snippet_tag = card.select_one(".result__snippet")
        if not link or not link.get("href"):
            continue

        url = normalize_whitespace(unwrap_duckduckgo_url(link["href"]))
        if "linkedin.com/in/" in url:
            continue

        results.append(
            SearchResult(
                title=normalize_whitespace(link.get_text(" ", strip=True)),
                url=url,
                snippet=normalize_whitespace(snippet_tag.get_text(" ", strip=True) if snippet_tag else ""),
                source_query=query,
                search_provider="duckduckgo",
                discovery_method="search",
            )
        )

    return ("ok" if results else "empty"), results


def search_bing_rss(session: requests.Session, query: str) -> tuple[str, list[SearchResult]]:
    response = session.get(
        SEARCH_ENDPOINT_BING,
        params={"q": query, "format": "rss"},
        timeout=REQUEST_TIMEOUT,
        headers={"Referer": "https://www.bing.com/", "User-Agent": USER_AGENT},
    )
    response.raise_for_status()

    if is_search_blocked("bing", response.status_code, response.text):
        return "blocked", []

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError:
        return "empty", []

    results = []
    for item in root.findall("./channel/item"):
        title = normalize_whitespace(item.findtext("title", default=""))
        url = normalize_whitespace(item.findtext("link", default=""))
        snippet = normalize_whitespace(item.findtext("description", default=""))
        if not url or "linkedin.com/in/" in url:
            continue
        results.append(
            SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                source_query=query,
                search_provider="bing_rss",
                discovery_method="search",
            )
        )

    return ("ok" if results else "empty"), results


def build_direct_probe_results(
    profile: LeadProfile,
    discovered_clues: dict[str, str],
    seen_urls: set[str],
) -> list[SearchResult]:
    results: list[SearchResult] = []
    handle_candidates = build_handle_candidates(profile, discovered_clues)
    for handle in handle_candidates:
        for platform_name, pattern in direct_probe_patterns():
            url = pattern.format(handle=handle)
            canonical = url.split("#")[0]
            if canonical in seen_urls:
                continue
            seen_urls.add(canonical)
            results.append(
                SearchResult(
                    title=f"Direct probe: {platform_name} {handle}",
                    url=url,
                    snippet="",
                    source_query=f"direct_probe:{handle}",
                    search_provider="direct_probe",
                    discovery_method="direct_probe",
                )
            )
            if len(results) >= MAX_DIRECT_PROBES:
                return results
    return results


def classify_platform(url: str) -> str:
    domain = urlparse(url).netloc.lower().replace("www.", "")
    for known_domain, label in SOCIAL_DOMAINS.items():
        if domain.endswith(known_domain):
            return label
    return domain or "Website"


def extract_json_ld_types(soup: BeautifulSoup) -> set[str]:
    types = set()
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text(" ", strip=True)
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue

        stack = payload if isinstance(payload, list) else [payload]
        while stack:
            current = stack.pop()
            if isinstance(current, dict):
                value = current.get("@type")
                if isinstance(value, str):
                    types.add(value.lower())
                elif isinstance(value, list):
                    types.update(str(item).lower() for item in value)
                for nested in current.values():
                    if isinstance(nested, (dict, list)):
                        stack.append(nested)
            elif isinstance(current, list):
                stack.extend(current)
    return types


def looks_like_known_profile_url(url: str) -> tuple[bool, str]:
    parsed = urlparse(url)
    domain = parsed.netloc.lower().replace("www.", "")
    segments = url_path_segments(url)
    first = segments[0].lower() if segments else ""
    reserved = PROFILE_RESERVED_SEGMENTS.get(domain, set())

    if not segments:
        return False, ""

    if domain in {"github.com", "x.com", "twitter.com", "instagram.com", "facebook.com", "about.me"}:
        if len(segments) == 1 and first not in reserved and "+" not in first:
            return True, "URL path matches a public profile format on the platform."
        return False, ""

    if domain == "medium.com":
        if len(segments) == 1 and segments[0].startswith("@"):
            return True, "Medium handle path matches a public profile format."
        return False, ""

    if domain == "dev.to":
        if len(segments) == 1 and first not in reserved:
            return True, "Dev.to username path matches a public profile format."
        return False, ""

    if domain == "youtube.com":
        if first.startswith("@"):
            return True, "YouTube handle path matches a public profile/channel format."
        if first in {"c", "channel", "user"} and len(segments) >= 2:
            return True, "YouTube channel path matches a public profile format."
        return False, ""

    if domain in {"wellfound.com", "angel.co"}:
        if first in {"u", "profile"} and len(segments) >= 2:
            return True, "Platform URL path looks like a user profile."
        return False, ""

    if domain == "crunchbase.com":
        if first == "person" and len(segments) >= 2:
            return True, "Crunchbase URL path looks like a person profile."
        return False, ""

    return False, ""


def classify_page(url: str, soup: BeautifulSoup | None, title: str, description: str, text_excerpt: str) -> tuple[str, bool, bool, list[str], list[str]]:
    domain = urlparse(url).netloc.lower().replace("www.", "")
    segments = url_path_segments(url)
    title_blob = normalize_whitespace(" ".join([title, description, text_excerpt])).lower()
    query = urlparse(url).query.lower()
    profile_signals: list[str] = []
    warning_signals: list[str] = []

    if query and any(key in query for key in {"q=", "query=", "search=", "keyword="}):
        warning_signals.append("URL contains search-style query parameters.")

    if segments:
        lowered_segments = [segment.lower() for segment in segments]
        if any(segment in SEARCH_LIKE_SEGMENTS for segment in lowered_segments):
            warning_signals.append("URL path looks like a search, directory, or topic page.")
        if any("+" in segment for segment in lowered_segments):
            warning_signals.append("URL path contains query-like tokens instead of a clean handle.")
        if any(segment in {"story", "news", "politics", "article"} for segment in lowered_segments):
            warning_signals.append("URL path looks like a news/article page.")
        if any(segment in {"author", "authors", "team", "bio", "speaker", "speakers"} for segment in lowered_segments):
            profile_signals.append("URL path looks like an author/team/bio profile page.")

    if any(term in title_blob for term in GENERIC_EVIDENCE_TERMS):
        warning_signals.append("Page title/description looks like generic search or language content, not a person profile.")
    if any(term in title_blob for term in ["biography", "news", "latest", "story", "politics", "article"]):
        warning_signals.append("Page appears to be a biography/news/article page rather than a social profile.")

    profile_like_url, url_reason = looks_like_known_profile_url(url)
    if profile_like_url:
        profile_signals.append(url_reason)

    schema_types = set()
    og_type = ""
    if soup is not None:
        schema_types = extract_json_ld_types(soup)
        if schema_types.intersection(PERSON_SCHEMA_TYPES):
            profile_signals.append("Structured data describes the page as a Person/ProfilePage.")

        og_type_tag = soup.find("meta", attrs={"property": "og:type"})
        if og_type_tag and og_type_tag.get("content"):
            og_type = og_type_tag.get("content", "").strip().lower()
            if og_type == "profile":
                profile_signals.append("Open Graph metadata marks this page as a profile.")

    profile_keyword_overlap = PROFILE_META_TERMS.intersection(set(tokenize(title_blob)))
    if profile_keyword_overlap and not warning_signals:
        profile_signals.append("Page content includes profile-style terms.")

    is_directory_like = bool(warning_signals) and not profile_like_url and not schema_types.intersection(PERSON_SCHEMA_TYPES)

    if profile_like_url:
        page_type = "social_profile" if domain in SOCIAL_DOMAINS else "profile_page"
    elif any(segment.lower() in {"author", "authors", "team", "bio", "speaker", "speakers"} for segment in segments) and (
        schema_types.intersection(PERSON_SCHEMA_TYPES) or len(profile_signals) > 0
    ):
        page_type = "profile_page"
    elif schema_types.intersection(PERSON_SCHEMA_TYPES) or og_type == "profile":
        page_type = "personal_site"
    elif is_directory_like:
        page_type = "directory_or_search"
    elif len(segments) > 1:
        page_type = "content_page"
    else:
        page_type = "unknown"

    if page_type == "personal_site" and (
        any(term in title_blob for term in ["biography", "news", "article", "story"])
        or any(segment.lower() in {"story", "news", "politics", "article"} for segment in segments)
    ):
        page_type = "content_page"

    is_profile_like = page_type in {"social_profile", "personal_site", "profile_page"}
    return page_type, is_profile_like, is_directory_like, profile_signals, warning_signals


def fetch_evidence(session: requests.Session, result: SearchResult) -> Evidence:
    domain = urlparse(result.url).netloc.lower().replace("www.", "")
    evidence = Evidence(
        url=result.url,
        title=result.title,
        description=result.snippet,
        text_excerpt="",
        domain=domain,
        platform=classify_platform(result.url),
    )

    try:
        html = safe_get(session, result.url)
    except requests.RequestException:
        return evidence

    soup = BeautifulSoup(html, "html.parser")
    page_title = normalize_whitespace(soup.title.get_text(" ", strip=True) if soup.title else "")
    description_tag = soup.find("meta", attrs={"name": "description"}) or soup.find(
        "meta", attrs={"property": "og:description"}
    )
    page_description = normalize_whitespace(description_tag.get("content", "") if description_tag else "")
    visible_text = normalize_whitespace(" ".join(p.get_text(" ", strip=True) for p in soup.find_all(["h1", "h2", "p"])[:12]))
    outgoing_links = []
    for link in soup.find_all("a", href=True):
        href = normalize_whitespace(link.get("href", ""))
        if href.startswith("http") and "linkedin.com/in/" not in href:
            outgoing_links.append(href)
        if len(outgoing_links) >= 12:
            break

    if page_title:
        evidence.title = page_title
    if page_description:
        evidence.description = page_description
    if visible_text:
        evidence.text_excerpt = visible_text[:700]
    if outgoing_links:
        evidence.outgoing_links = unique_preserve_order(outgoing_links)

    (
        evidence.page_type,
        evidence.is_profile_like,
        evidence.is_directory_like,
        evidence.profile_signals,
        evidence.warning_signals,
    ) = classify_page(result.url, soup, evidence.title, evidence.description, evidence.text_excerpt)

    return evidence


def heuristic_extract_candidate_facts(profile: LeadProfile, evidence: Evidence) -> CandidateFacts:
    handle = evidence.url.rstrip("/").split("/")[-1]
    title = normalize_whitespace(evidence.title)
    description = normalize_whitespace(evidence.description)
    text_excerpt = normalize_whitespace(evidence.text_excerpt)
    combined = " ".join(part for part in [title, description, text_excerpt] if part)

    display_name = ""
    if title:
        display_name = normalize_whitespace(re.split(r"[|\-–]", title, maxsplit=1)[0])
        if len(display_name.split()) < 2:
            display_name = ""

    facts = CandidateFacts(
        display_name=display_name,
        handle=handle,
        bio=description or text_excerpt[:220],
        links=unique_preserve_order([evidence.url] + evidence.outgoing_links[:6]),
        evidence_snippets=[snippet for snippet in [title, description, text_excerpt[:220]] if snippet],
    )

    company_overlap = sorted(set(tokenize(profile.company)).intersection(set(tokenize(combined))))
    if company_overlap:
        facts.company = " ".join(company_overlap)
        facts.extraction_notes.append("Company clue inferred from overlap with candidate page text.")

    location_overlap = sorted(set(tokenize(profile.location)).intersection(set(tokenize(combined))))
    if location_overlap:
        facts.location = " ".join(location_overlap)
        facts.extraction_notes.append("Location clue inferred from overlap with candidate page text.")

    if evidence.is_profile_like:
        facts.extraction_notes.append("Candidate page is classified as profile-like.")
    if evidence.warning_signals:
        facts.extraction_notes.extend(evidence.warning_signals[:1])

    mentioned_handles = re.findall(r"@([A-Za-z0-9._]{3,30})", " ".join([description, text_excerpt]))
    if mentioned_handles and not facts.handle:
        facts.handle = mentioned_handles[0]

    return facts


def llm_extract_candidate_facts(
    session: requests.Session,
    context: ExecutionContext,
    profile: LeadProfile,
    evidence: Evidence,
    heuristic_facts: CandidateFacts,
) -> CandidateFacts:
    if context.actual_mode != "agent" or not context.active_model:
        return heuristic_facts

    prompt = textwrap.dedent(
        f"""
        Normalize the candidate page into structured facts using only the supplied evidence.
        Do not infer facts that are not clearly supported by the evidence.

        Lead seed:
        - name: {profile.full_name or 'unknown'}
        - headline: {profile.headline or 'unknown'}
        - company: {profile.company or 'unknown'}
        - location: {profile.location or 'unknown'}

        Candidate evidence:
        - url: {evidence.url}
        - platform: {evidence.platform}
        - page_type: {evidence.page_type}
        - title: {evidence.title}
        - description: {evidence.description}
        - text_excerpt: {evidence.text_excerpt}

        Heuristic extraction:
        {json.dumps(asdict(heuristic_facts), ensure_ascii=False)}

        Return JSON only:
        {{
          "display_name": "",
          "handle": "",
          "bio": "",
          "company": "",
          "location": "",
          "links": ["..."],
          "evidence_snippets": ["..."],
          "extraction_notes": ["..."]
        }}
        """
    ).strip()
    system_prompt = "You extract structured facts from web evidence. Return only strict JSON."
    parsed = groq_chat_json(session, context.active_model, system_prompt, prompt, context.trace, "evidence_extractor")
    if not parsed:
        return heuristic_facts

    facts = CandidateFacts(
        display_name=normalize_whitespace(str(parsed.get("display_name", ""))),
        handle=normalize_whitespace(str(parsed.get("handle", ""))) or heuristic_facts.handle,
        bio=normalize_whitespace(str(parsed.get("bio", ""))) or heuristic_facts.bio,
        company=normalize_whitespace(str(parsed.get("company", ""))) or heuristic_facts.company,
        location=normalize_whitespace(str(parsed.get("location", ""))) or heuristic_facts.location,
        links=unique_preserve_order([str(item) for item in parsed.get("links", []) if normalize_whitespace(str(item))]) or heuristic_facts.links,
        evidence_snippets=unique_preserve_order(
            [normalize_whitespace(str(item)) for item in parsed.get("evidence_snippets", []) if normalize_whitespace(str(item))]
        )
        or heuristic_facts.evidence_snippets,
        extraction_notes=unique_preserve_order(
            [normalize_whitespace(str(item)) for item in parsed.get("extraction_notes", []) if normalize_whitespace(str(item))]
        )
        or heuristic_facts.extraction_notes,
        normalized_by="groq",
    )
    return facts


def build_identity_signal_map(profile: LeadProfile, evidence: Evidence, facts: CandidateFacts) -> dict[str, str]:
    signals: dict[str, str] = {}
    combined_text = " ".join(
        [
            evidence.title,
            evidence.description,
            evidence.text_excerpt,
            facts.display_name,
            facts.bio,
            facts.company,
            facts.location,
            facts.handle,
        ]
    ).lower()
    combined_tokens = set(tokenize(combined_text))

    full_name = normalize_name(profile.full_name)
    if full_name and full_name in normalize_name(" ".join([facts.display_name, evidence.title, evidence.description])):
        signals["full_name"] = "Full name aligns between LinkedIn seed and candidate profile."
    else:
        first = normalize_name(profile.first_name)
        last = normalize_name(profile.last_name)
        if first and last and first in combined_tokens and last in combined_tokens:
            signals["name_parts"] = "First and last name both appear on the candidate page."

    handle_tokens = set(tokenize(facts.handle.replace("-", " ").replace("_", " ")))
    slug_tokens = set(tokenize(profile.slug.replace("-", " ")))
    if evidence.is_profile_like and handle_tokens and slug_tokens.intersection(handle_tokens):
        signals["handle"] = "Candidate handle overlaps with the LinkedIn slug."

    company_overlap = set(tokenize(profile.company)).intersection(set(tokenize(" ".join([facts.company, facts.bio, evidence.description]))))
    if company_overlap:
        signals["company"] = f"Shared company clue: {', '.join(sorted(company_overlap))}."

    role_overlap = {token for token in tokenize(profile.headline) if token not in ROLE_STOPWORDS}.intersection(
        set(tokenize(" ".join([facts.bio, evidence.description, evidence.title])))
    )
    if role_overlap:
        signals["role"] = f"Shared role/headline clue: {', '.join(sorted(role_overlap))}."

    location_overlap = set(tokenize(profile.location)).intersection(set(tokenize(" ".join([facts.location, evidence.description, evidence.text_excerpt]))))
    if location_overlap:
        signals["location"] = f"Shared location clue: {', '.join(sorted(location_overlap))}."

    return signals


def is_acceptance_eligible_profile(match: MatchResult) -> bool:
    return match.evidence.page_type in {"social_profile", "profile_page"}


def compute_match(profile: LeadProfile, evidence: Evidence) -> MatchResult:
    reasons = []
    score = 0

    name = normalize_name(profile.full_name)
    first = normalize_name(profile.first_name)
    last = normalize_name(profile.last_name)
    company_tokens = set(tokenize(profile.company))
    role_tokens = {token for token in tokenize(profile.headline) if token not in ROLE_STOPWORDS}
    location_tokens = set(tokenize(profile.location))
    slug_tokens = set(tokenize(profile.slug.replace("-", " ")))

    haystack = normalize_name(" ".join([evidence.title, evidence.description, evidence.text_excerpt, evidence.url]))
    haystack_tokens = set(tokenize(haystack))

    if evidence.is_profile_like:
        score += 25
        reasons.extend(evidence.profile_signals[:2])
    elif evidence.page_type == "content_page":
        score += 5
        reasons.append("Page may provide indirect evidence, but it is not clearly a profile page.")

    if name and name in haystack:
        score += 25 if evidence.is_profile_like else 10
        reasons.append("Exact full-name match found in title/description/page text.")
    elif first and last and first in haystack_tokens and last in haystack_tokens:
        score += 20 if evidence.is_profile_like else 8
        reasons.append("First and last name both appear on the candidate page.")
    elif first and first in haystack_tokens:
        score += 5
        reasons.append("First name appears on the candidate page.")

    handle = evidence.url.rstrip("/").split("/")[-1]
    handle_tokens = set(tokenize(handle.replace("-", " ").replace("_", " ")))
    if evidence.is_profile_like and slug_tokens and slug_tokens.intersection(handle_tokens):
        overlap = len(slug_tokens.intersection(handle_tokens))
        score += min(15, overlap * 5)
        reasons.append("Username/handle overlaps with the LinkedIn slug.")

    if company_tokens:
        overlap = company_tokens.intersection(haystack_tokens)
        if overlap:
            score += min(20, len(overlap) * 8)
            reasons.append(f"Company overlap detected: {', '.join(sorted(overlap))}.")

    if role_tokens:
        overlap = role_tokens.intersection(haystack_tokens)
        if overlap:
            score += min(15, len(overlap) * 5)
            reasons.append(f"Role/headline overlap detected: {', '.join(sorted(overlap))}.")

    if location_tokens:
        overlap = location_tokens.intersection(haystack_tokens)
        if overlap:
            score += min(10, len(overlap) * 5)
            reasons.append(f"Location overlap detected: {', '.join(sorted(overlap))}.")

    if evidence.is_directory_like:
        score -= 40
        reasons.extend(evidence.warning_signals[:2])
    elif not evidence.is_profile_like and evidence.page_type == "unknown":
        score -= 15
        reasons.append("Page is not clearly a public profile or personal site.")

    conflicting_name = False
    if profile.last_name and profile.last_name.lower() not in evidence.title.lower() and profile.full_name:
        title_tokens = set(tokenize(evidence.title))
        if title_tokens and profile.first_name.lower() in title_tokens and profile.last_name.lower() not in title_tokens:
            conflicting_name = True

    if conflicting_name:
        score -= 20
        reasons.append("Candidate title mentions a partial/conflicting name.")

    score = max(0, min(100, score))
    if score >= 70:
        label = "strong"
    elif score >= 50:
        label = "probable"
    elif score >= 30:
        label = "weak"
    else:
        label = "unlikely"

    return MatchResult(
        candidate_url=evidence.url,
        platform=evidence.platform,
        confidence_score=score,
        confidence_label=label,
        reasons=reasons or ["Only weak indirect evidence was found."],
        evidence=evidence,
    )


def heuristic_decision_for_match(profile: LeadProfile, match: MatchResult, facts: CandidateFacts) -> CandidateDecision:
    signals = build_identity_signal_map(profile, match.evidence, facts)
    citations = unique_preserve_order(
        [
            f"URL: {match.candidate_url}",
            f"Title: {match.evidence.title}" if match.evidence.title else "",
            f"Description: {match.evidence.description[:180]}" if match.evidence.description else "",
        ]
    )

    if match.evidence.page_type == "content_page" and len(signals) >= 2 and match.confidence_score >= 25:
        return CandidateDecision(
            status="manual_review",
            rationale="Candidate is a biography/news/content page with real identity overlap, so it should be reviewed manually rather than accepted.",
            matched_signals=list(signals.values()),
            citations=citations,
            review_notes=match.reasons[:3],
            decided_by="heuristic",
        )

    if match.evidence.page_type == "personal_site":
        return CandidateDecision(
            status="manual_review",
            rationale="Candidate looks like a person-related page, but only explicit profile URLs can be accepted automatically.",
            matched_signals=list(signals.values()),
            citations=citations,
            review_notes=match.reasons[:3],
            decided_by="heuristic",
        )

    if not match.evidence.is_profile_like and match.evidence.page_type != "personal_site":
        return CandidateDecision(
            status="rejected",
            rationale="Candidate is not a profile-like page or personal site.",
            matched_signals=list(signals.values()),
            citations=citations,
            review_notes=match.reasons[:3],
            decided_by="heuristic",
        )

    if is_acceptance_eligible_profile(match) and match.confidence_score >= MIN_ACCEPTED_PROFILE_SCORE and len(signals) >= 2:
        return CandidateDecision(
            status="accepted",
            rationale="Candidate is a real profile URL and passed the identity-evidence gate.",
            matched_signals=list(signals.values()),
            citations=citations,
            review_notes=match.reasons[:3],
            decided_by="heuristic",
        )

    if match.confidence_score >= 25 and len(signals) >= 1:
        return CandidateDecision(
            status="manual_review",
            rationale="Candidate has some identity overlap but not enough evidence for acceptance.",
            matched_signals=list(signals.values()),
            citations=citations,
            review_notes=match.reasons[:3],
            decided_by="heuristic",
        )

    return CandidateDecision(
        status="rejected",
        rationale="Candidate does not meet the minimum identity-evidence floor.",
        matched_signals=list(signals.values()),
        citations=citations,
        review_notes=match.reasons[:3],
        decided_by="heuristic",
    )


def llm_verify_candidate(
    session: requests.Session,
    context: ExecutionContext,
    profile: LeadProfile,
    match: MatchResult,
    facts: CandidateFacts,
    heuristic_decision: CandidateDecision,
) -> CandidateDecision:
    if context.actual_mode != "agent" or not context.active_model:
        return heuristic_decision

    signals = build_identity_signal_map(profile, match.evidence, facts)
    prompt = textwrap.dedent(
        f"""
        Decide whether this candidate belongs to the same person as the LinkedIn lead.
        Use only the supplied evidence. Be conservative. Prefer rejected or manual_review over accepted unless identity evidence is strong.
        Never accept a candidate that is not a profile-like page or personal site.

        Lead seed:
        {json.dumps(asdict(profile), ensure_ascii=False)}

        Candidate heuristic match:
        {json.dumps(asdict(match), ensure_ascii=False)}

        Candidate structured facts:
        {json.dumps(asdict(facts), ensure_ascii=False)}

        Heuristic signal map:
        {json.dumps(signals, ensure_ascii=False)}

        Heuristic decision:
        {json.dumps(asdict(heuristic_decision), ensure_ascii=False)}

        Return JSON only:
        {{
          "status": "accepted|rejected|manual_review",
          "rationale": "",
          "matched_signals": ["..."],
          "citations": ["..."],
          "review_notes": ["..."]
        }}
        """
    ).strip()
    system_prompt = "You are an identity-verification agent for lead enrichment. Return only strict JSON."
    parsed = groq_chat_json(session, context.active_model, system_prompt, prompt, context.trace, "identity_verifier")
    if not parsed:
        return heuristic_decision

    llm_decision = CandidateDecision(
        status=normalize_whitespace(str(parsed.get("status", ""))).lower() or heuristic_decision.status,
        rationale=normalize_whitespace(str(parsed.get("rationale", ""))) or heuristic_decision.rationale,
        matched_signals=unique_preserve_order(
            [normalize_whitespace(str(item)) for item in parsed.get("matched_signals", []) if normalize_whitespace(str(item))]
        )
        or heuristic_decision.matched_signals,
        citations=unique_preserve_order(
            [normalize_whitespace(str(item)) for item in parsed.get("citations", []) if normalize_whitespace(str(item))]
        )
        or heuristic_decision.citations,
        review_notes=unique_preserve_order(
            [normalize_whitespace(str(item)) for item in parsed.get("review_notes", []) if normalize_whitespace(str(item))]
        )
        or heuristic_decision.review_notes,
        decided_by="groq",
    )

    minimum_gates_met = (
        is_acceptance_eligible_profile(match)
        and match.evidence.is_profile_like
        and
        match.confidence_score >= MIN_ACCEPTED_PROFILE_SCORE
        and len(build_identity_signal_map(profile, match.evidence, facts)) >= 2
    )
    if llm_decision.status == "accepted" and not minimum_gates_met:
        llm_decision.status = "manual_review" if match.evidence.is_profile_like else "rejected"
        llm_decision.review_notes.append("Guardrail prevented promotion to accepted because minimum heuristic gates were not met.")

    if llm_decision.status not in {"accepted", "rejected", "manual_review"}:
        return heuristic_decision

    return llm_decision


def filter_reportable_assessments(assessments: list[CandidateAssessment]) -> list[CandidateAssessment]:
    return [
        assessment
        for assessment in assessments
        if assessment.match.confidence_score >= MIN_REPORT_SCORE and not assessment.match.evidence.is_directory_like
    ]


def search_candidates_for_queries(
    session: requests.Session,
    queries: list[str],
    trace: list[dict[str, Any]],
    seen_urls: set[str],
) -> list[SearchResult]:
    all_results: list[SearchResult] = []
    for query in queries:
        provider_used = "duckduckgo"
        status = "empty"
        results: list[SearchResult] = []

        time.sleep(SEARCH_DELAY_SECONDS)
        try:
            status, results = search_duckduckgo(session, query)
        except requests.RequestException as exc:
            add_trace(trace, "search_tool", "DuckDuckGo query failed.", {"query": query, "provider": "duckduckgo", "error": str(exc)})
            status = "blocked"

        if status == "blocked":
            provider_used = "bing_rss"
            time.sleep(SEARCH_DELAY_SECONDS)
            try:
                status, results = search_bing_rss(session, query)
            except requests.RequestException as exc:
                add_trace(trace, "search_tool", "Bing RSS query failed.", {"query": query, "provider": "bing_rss", "error": str(exc)})
                status = "blocked"
                results = []

        add_trace(
            trace,
            "search_tool",
            "Executed web search query.",
            {
                "query": query,
                "search_provider": provider_used,
                "search_status": status,
                "result_count": len(results),
            },
        )
        for result in results:
            parsed = urlparse(result.url)
            if not parsed.scheme.startswith("http"):
                continue
            canonical = result.url.split("#")[0]
            if canonical in seen_urls:
                continue
            seen_urls.add(canonical)
            all_results.append(result)
            if len(all_results) >= MAX_CANDIDATES:
                return all_results
    return all_results


def collect_round_candidates(
    profile: LeadProfile,
    session: requests.Session,
    context: ExecutionContext,
    queries: list[str],
    discovered_clues: dict[str, str],
    seen_urls: set[str],
) -> list[SearchResult]:
    candidates = search_candidates_for_queries(session, queries, context.trace, seen_urls)
    direct_probe_results = build_direct_probe_results(profile, discovered_clues, seen_urls)
    if direct_probe_results:
        add_trace(
            context.trace,
            "direct_probe",
            "Generated direct profile probes.",
            {"count": len(direct_probe_results), "sample_urls": [item.url for item in direct_probe_results[:6]]},
        )
        candidates.extend(direct_probe_results)

    return candidates[:MAX_CANDIDATES]


def assess_candidate(
    profile: LeadProfile,
    candidate: SearchResult,
    session: requests.Session,
    context: ExecutionContext,
) -> CandidateAssessment:
    time.sleep(SEARCH_DELAY_SECONDS)
    evidence = fetch_evidence(session, candidate)
    match = compute_match(profile, evidence)
    heuristic_facts = heuristic_extract_candidate_facts(profile, evidence)
    facts = llm_extract_candidate_facts(session, context, profile, evidence, heuristic_facts)
    heuristic_decision = heuristic_decision_for_match(profile, match, facts)
    final_decision = llm_verify_candidate(session, context, profile, match, facts, heuristic_decision)
    assessment = CandidateAssessment(match=match, facts=facts, decision=final_decision)
    add_trace(
        context.trace,
        "candidate_assessment",
        "Assessed candidate page.",
        {
            "url": candidate.url,
            "platform": match.platform,
            "score": match.confidence_score,
            "page_type": match.evidence.page_type,
            "decision": final_decision.status,
        },
    )
    return assessment


def collect_discovered_clues(assessments: list[CandidateAssessment]) -> dict[str, str]:
    clues = {"handle": "", "company": "", "role": "", "location": "", "website": ""}
    prioritized = sorted(assessments, key=lambda item: item.match.confidence_score, reverse=True)
    for assessment in prioritized:
        if assessment.decision.status == "rejected":
            continue
        if not clues["handle"] and assessment.facts.handle and is_acceptance_eligible_profile(assessment.match):
            clues["handle"] = assessment.facts.handle
        if not clues["company"] and assessment.facts.company:
            clues["company"] = assessment.facts.company
        if not clues["role"] and assessment.facts.bio:
            clues["role"] = assessment.facts.bio[:80]
        if not clues["location"] and assessment.facts.location:
            clues["location"] = assessment.facts.location
        if not clues["website"]:
            for link in assessment.facts.links:
                domain = urlparse(link).netloc.lower().replace("www.", "")
                if domain and domain not in SOCIAL_DOMAINS and "linkedin.com" not in domain:
                    clues["website"] = link
                    break
    return {key: value for key, value in clues.items() if value}


def merge_assessment_evidence(
    profile: LeadProfile,
    accepted: list[CandidateAssessment],
    manual_review: list[CandidateAssessment],
) -> dict[str, Any]:
    relevant = accepted + manual_review
    handles = unique_preserve_order(
        assessment.facts.handle for assessment in relevant if normalize_whitespace(assessment.facts.handle)
    )
    websites = unique_preserve_order(
        link
        for assessment in relevant
        for link in assessment.facts.links
        if urlparse(link).scheme.startswith("http") and "linkedin.com/in/" not in link
    )
    platforms = unique_preserve_order(assessment.match.platform for assessment in accepted)
    corroborating_platforms = unique_preserve_order(assessment.match.platform for assessment in manual_review)

    companies = unique_preserve_order(
        company
        for company in [profile.company]
        + [assessment.facts.company for assessment in relevant]
        if normalize_whitespace(company)
    )
    locations = unique_preserve_order(
        location
        for location in [profile.location]
        + [assessment.facts.location for assessment in relevant]
        if normalize_whitespace(location)
    )

    themes = set()
    for assessment in relevant:
        for token in tokenize(" ".join([assessment.facts.bio, assessment.match.evidence.description, assessment.match.evidence.title])):
            if token not in ROLE_STOPWORDS and len(token) > 3:
                themes.add(token)

    return {
        "accepted_platforms": platforms,
        "corroborating_platforms": corroborating_platforms,
        "handles": handles[:8],
        "websites": websites[:12],
        "companies": companies[:6],
        "locations": locations[:6],
        "themes": sorted(themes)[:20],
    }


def build_identity_decision_summary(profile: LeadProfile, accepted: list[CandidateAssessment], manual_review: list[CandidateAssessment]) -> str:
    if accepted:
        platforms = ", ".join(assessment.match.platform for assessment in accepted[:3])
        return (
            f"The lead was enriched with {len(accepted)} accepted external profile match(es). "
            f"Verified evidence was found on {platforms}."
        )
    if manual_review:
        return (
            "Some candidate profiles show partial overlap with the LinkedIn lead, "
            "but the evidence is not strong enough for confident acceptance."
        )
    return (
        "No reliable external profiles or personal websites were confidently verified beyond the LinkedIn seed."
    )


def build_discovery_notes(trace: list[dict[str, Any]], assessments: list[CandidateAssessment]) -> list[str]:
    search_events = [item for item in trace if item.get("step") == "search_tool"]
    notes = []
    if not search_events:
        return notes

    blocked = [item for item in search_events if item.get("data", {}).get("search_status") == "blocked"]
    successful = [item for item in search_events if item.get("data", {}).get("search_status") == "ok"]
    empty = [item for item in search_events if item.get("data", {}).get("search_status") == "empty"]

    if blocked and not successful:
        notes.append("Search discovery was blocked or degraded by the search providers for all attempted queries.")
    elif successful and not assessments:
        notes.append("Search providers returned results, but no candidate pages passed the initial collection filters.")
    elif empty and not successful:
        notes.append("Search providers responded, but the attempted queries returned no usable search results.")

    return notes


def build_outreach_observation(profile: LeadProfile, assessment: CandidateAssessment) -> str:
    headline_tokens = set(tokenize(profile.headline))
    themes = []

    if headline_tokens.intersection({"evp", "executive", "founder", "organization", "business"}):
        themes.append("business")
    if headline_tokens.intersection({"news", "media", "publisher"}):
        themes.append("media")
    if headline_tokens.intersection({"outdoorsman", "outdoor"}):
        themes.append("the outdoors")
    if headline_tokens.intersection({"dad", "father"}):
        themes.append("family")

    if themes:
        if len(themes) == 1:
            theme_text = themes[0]
        elif len(themes) == 2:
            theme_text = f"{themes[0]} and {themes[1]}"
        else:
            theme_text = ", ".join(themes[:-1]) + f", and {themes[-1]}"
        return f"the mix of {theme_text} in your public profile stood out"

    bio = normalize_whitespace(assessment.facts.bio or assessment.match.evidence.description)
    if assessment.match.platform == "Instagram" and " on instagram:" in bio.lower():
        tail = bio.split(" on Instagram:", 1)[-1] if " on Instagram:" in bio else bio
        tail = tail.strip().strip('"')
        if tail:
            return f'your Instagram bio caught my attention, especially "{tail[:120].rstrip(" .")}"'

    return f"your public presence on {assessment.match.platform} stood out"


def build_seed_based_outreach(profile: LeadProfile, manual_review: list[CandidateAssessment]) -> str:
    first_name = profile.first_name or (profile.full_name.split()[0] if profile.full_name else "there")
    headline_phrase = ""
    if profile.headline:
        lowered = profile.headline.rstrip(".")
        if lowered.lower().startswith("leader in "):
            headline_phrase = f"your focus on {lowered[10:]}"
        else:
            headline_phrase = f"your background in {lowered[0].lower() + lowered[1:]}"

    if headline_phrase and profile.company:
        opening = f"I came across your LinkedIn profile, and {headline_phrase} at {profile.company} stood out."
    elif headline_phrase:
        opening = f"I came across your LinkedIn profile, and {headline_phrase} stood out."
    elif profile.company:
        opening = f"I came across your LinkedIn profile, and the work you are doing at {profile.company} stood out."
    else:
        opening = "I came across your LinkedIn profile and wanted to reach out with a specific idea."

    location_line = f" I also noticed you are based in {profile.location}." if profile.location else ""

    pitch_anchor = profile.company or "what you are building"
    return textwrap.dedent(
        f"""
        Hi {first_name},

        {opening}{location_line}
        I have a concise idea that seems relevant to {pitch_anchor}, and I can send it as a short note if useful.

        Best,
        [Your Name]
        """
    ).strip()


def build_fallback_summary(profile: LeadProfile, accepted: list[CandidateAssessment], manual_review: list[CandidateAssessment]) -> tuple[str, str]:
    merged = merge_assessment_evidence(profile, accepted, manual_review)
    if not accepted:
        summary = (
            "Public-web enrichment was inconclusive. No accepted external profile matches were verified beyond the LinkedIn seed."
        )
        if merged["corroborating_platforms"]:
            summary += " Supporting public pages were found but require manual review."
        outreach = build_seed_based_outreach(profile, manual_review)
        return summary, outreach

    profile_matches = [assessment for assessment in accepted if is_acceptance_eligible_profile(assessment.match)]
    summary_targets = profile_matches or accepted

    fact_lines = []
    for assessment in summary_targets[:2]:
        fact = assessment.facts.bio or assessment.match.evidence.description or assessment.match.evidence.title
        fact_lines.append(f"{assessment.match.platform}: {normalize_whitespace(fact)[:160]}")

    summary = (
        f"{profile.full_name or 'The lead'} appears to have verified public presence on "
        f"{', '.join(assessment.match.platform for assessment in summary_targets[:3])}. "
        f"Top evidence: {' | '.join(fact_lines)}"
    )
    if merged["corroborating_platforms"]:
        summary += f" Corroborating public pages were also found on {', '.join(merged['corroborating_platforms'][:4])}."

    first_name = profile.first_name or (profile.full_name.split()[0] if profile.full_name else "there")
    best_match = summary_targets[0]
    observation = build_outreach_observation(profile, best_match)
    opener = f"I came across your {best_match.match.platform} and {observation}."

    second_point = ""
    if len(summary_targets) > 1:
        second_point = f" I also found a consistent public profile on {summary_targets[1].match.platform}."

    outreach = textwrap.dedent(
        f"""
        Hi {first_name},

        {opener}{second_point}
        I have a specific idea that may fit that profile, and I would keep it brief.
        If you are open to it, I can send a short note with the idea and why I thought of you specifically.

        Best,
        [Your Name]
        """
    ).strip()
    return summary, outreach


def llm_synthesize_outputs(
    session: requests.Session,
    context: ExecutionContext,
    profile: LeadProfile,
    accepted: list[CandidateAssessment],
    manual_review: list[CandidateAssessment],
) -> tuple[str, str]:
    fallback_summary, fallback_outreach = build_fallback_summary(profile, accepted, manual_review)
    if context.actual_mode != "agent" or not context.active_model or not accepted:
        return fallback_summary, fallback_outreach

    accepted_payload = [asdict(assessment) for assessment in accepted[:MAX_REPORT_MATCHES]]
    prompt = textwrap.dedent(
        f"""
        Create a grounded lead summary and a short outreach draft using only accepted evidence from real profile URLs.
        Do not mention facts that are not present in the evidence.
        The outreach should sound human, concise, and specific.
        Avoid robotic phrases such as "your experience looks relevant" or "I think this could be worth a conversation."
        Use one concrete public detail, then make a soft ask for permission to send a short idea or note.
        If the evidence is too thin for meaningful personalization, keep the outreach cautious.

        Lead seed:
        {json.dumps(asdict(profile), ensure_ascii=False)}

        Accepted matches:
        {json.dumps(accepted_payload, ensure_ascii=False)}

        Return JSON only:
        {{
          "lead_summary": "",
          "outreach_message": ""
        }}
        """
    ).strip()
    system_prompt = "You synthesize grounded lead-enrichment reports. Return only strict JSON."
    parsed = groq_chat_json(session, context.active_model, system_prompt, prompt, context.trace, "report_synthesizer")
    if not parsed:
        return fallback_summary, fallback_outreach

    lead_summary = normalize_whitespace(str(parsed.get("lead_summary", ""))) or fallback_summary
    outreach_message = str(parsed.get("outreach_message", "")).strip() or fallback_outreach
    return lead_summary, outreach_message


def run_enrichment_pipeline(
    profile: LeadProfile,
    session: requests.Session,
    context: ExecutionContext,
) -> tuple[list[CandidateAssessment], list[CandidateAssessment], list[CandidateAssessment], str, str, str]:
    seen_urls: set[str] = set()
    all_assessments: list[CandidateAssessment] = []
    discovered_clues: dict[str, str] = {}

    for round_index in range(AGENT_MAX_SEARCH_ROUNDS):
        queries = plan_search_queries(profile, context, session, round_index, discovered_clues)
        if not queries:
            break

        for query in queries:
            if query not in profile.search_queries:
                profile.search_queries.append(query)

        candidates = collect_round_candidates(profile, session, context, queries, discovered_clues, seen_urls)
        for candidate in candidates:
            assessment = assess_candidate(profile, candidate, session, context)
            all_assessments.append(assessment)

            accepted_so_far = [item for item in all_assessments if item.decision.status == "accepted"]
            if len(accepted_so_far) >= STOP_AFTER_ACCEPTED_MATCHES:
                break

        accepted_so_far = [item for item in all_assessments if item.decision.status == "accepted"]
        if len(accepted_so_far) >= STOP_AFTER_ACCEPTED_MATCHES:
            add_trace(context.trace, "search_loop", "Stopped early after enough accepted matches were found.", {"accepted_count": len(accepted_so_far)})
            break

        discovered_clues = collect_discovered_clues(all_assessments)
        add_trace(context.trace, "search_loop", "Prepared discovered clues for the next round.", {"round": round_index + 1, "clues": discovered_clues})
        if round_index == AGENT_MAX_SEARCH_ROUNDS - 1 or not discovered_clues:
            break

    accepted = sorted(
        [item for item in all_assessments if item.decision.status == "accepted"],
        key=lambda item: item.match.confidence_score,
        reverse=True,
    )
    manual_review = sorted(
        [item for item in all_assessments if item.decision.status == "manual_review"],
        key=lambda item: item.match.confidence_score,
        reverse=True,
    )
    rejected = sorted(
        [item for item in all_assessments if item.decision.status == "rejected"],
        key=lambda item: item.match.confidence_score,
        reverse=True,
    )

    lead_summary, outreach_message = llm_synthesize_outputs(session, context, profile, accepted, manual_review)
    identity_decision = build_identity_decision_summary(profile, accepted, manual_review)
    return accepted, manual_review, rejected, lead_summary, outreach_message, identity_decision


def render_markdown_report(
    profile: LeadProfile,
    context: ExecutionContext,
    accepted: list[CandidateAssessment],
    manual_review: list[CandidateAssessment],
    rejected: list[CandidateAssessment],
    lead_summary: str,
    outreach_message: str,
    identity_decision: str,
) -> str:
    discovery_notes = build_discovery_notes(context.trace, accepted + manual_review + rejected)
    merged = merge_assessment_evidence(profile, accepted, manual_review)
    lines = [
        f"# Lead Enrichment Report: {profile.full_name or profile.slug or 'Unknown Lead'}",
        "",
        "## Execution",
        f"- Requested mode: {context.requested_mode}",
        f"- Actual mode: {context.actual_mode}",
        f"- Requested model: {context.requested_model}",
        f"- Active model: {context.active_model or 'none'}",
        f"- Groq available: {'yes' if context.groq_available else 'no'}",
    ]
    if context.fallback_reason:
        lines.append(f"- Fallback reason: {context.fallback_reason}")

    lines.extend(
        [
            "",
            "## LinkedIn Seed",
            f"- URL: {profile.linkedin_url}",
            f"- Name: {profile.full_name or 'Not confidently extracted'}",
            f"- Headline: {profile.headline or 'Not confidently extracted'}",
            f"- Company: {profile.company or 'Not confidently extracted'}",
            f"- Location: {profile.location or 'Not confidently extracted'}",
            "",
            "## Lead Summary",
            lead_summary,
            "",
            "## Identity Decision",
            identity_decision,
            "",
            "## Discovery Notes",
        ]
    )
    if not discovery_notes:
        lines.append("- Search discovery completed without provider-level blocking notes.")
    else:
        lines.extend(f"- {note}" for note in discovery_notes)

    lines.extend(
        [
            "",
            "## Accepted Matches",
        ]
    )

    if not accepted:
        lines.append("- No accepted external matches.")
    else:
        for assessment in accepted[:MAX_REPORT_MATCHES]:
            lines.extend(
                [
                    f"- {assessment.match.platform}: {assessment.match.candidate_url}",
                    f"  Confidence: {assessment.match.confidence_label} ({assessment.match.confidence_score}/100)",
                    f"  Page type: {assessment.match.evidence.page_type}",
                    f"  Decision: {assessment.decision.status} via {assessment.decision.decided_by}",
                    f"  Facts: name={assessment.facts.display_name or 'n/a'} | handle={assessment.facts.handle or 'n/a'} | company={assessment.facts.company or 'n/a'} | location={assessment.facts.location or 'n/a'}",
                    f"  Why accepted: {assessment.decision.rationale}",
                    f"  Signals: {'; '.join(assessment.decision.matched_signals) or 'n/a'}",
                    f"  Citations: {' | '.join(assessment.decision.citations) or 'n/a'}",
                ]
            )

    lines.extend(["", "## Manual Review Candidates"])
    if not manual_review:
        lines.append("- No manual-review candidates.")
    else:
        for assessment in manual_review[:MAX_REPORT_MATCHES]:
            lines.extend(
                [
                    f"- {assessment.match.platform}: {assessment.match.candidate_url}",
                    f"  Confidence: {assessment.match.confidence_label} ({assessment.match.confidence_score}/100)",
                    f"  Why held for review: {assessment.decision.rationale}",
                    f"  Signals: {'; '.join(assessment.decision.matched_signals) or 'n/a'}",
                ]
            )

    lines.extend(["", "## Merged Evidence"])
    lines.append(f"- Accepted platforms: {', '.join(merged['accepted_platforms']) or 'n/a'}")
    lines.append(f"- Corroborating platforms/pages: {', '.join(merged['corroborating_platforms']) or 'n/a'}")
    lines.append(f"- Handles: {', '.join(merged['handles']) or 'n/a'}")
    lines.append(f"- Websites: {', '.join(merged['websites'][:6]) or 'n/a'}")
    lines.append(f"- Companies: {', '.join(merged['companies']) or 'n/a'}")
    lines.append(f"- Locations: {', '.join(merged['locations']) or 'n/a'}")

    lines.extend(["", "## Rejected Candidates"])
    if not rejected:
        lines.append("- No rejected candidates were recorded.")
    else:
        for assessment in rejected[:MAX_REPORT_MATCHES]:
            lines.extend(
                [
                    f"- {assessment.match.platform}: {assessment.match.candidate_url}",
                    f"  Confidence: {assessment.match.confidence_label} ({assessment.match.confidence_score}/100)",
                    f"  Why rejected: {assessment.decision.rationale}",
                ]
            )

    lines.extend(
        [
            "",
            "## Outreach Message",
            outreach_message,
            "",
            "## Source Notes",
        ]
    )
    lines.extend(f"- {note}" for note in profile.source_notes)
    lines.extend(f"- Search query used: {query}" for query in profile.search_queries)
    return "\n".join(lines)


def build_output_payload(
    profile: LeadProfile,
    context: ExecutionContext,
    accepted: list[CandidateAssessment],
    manual_review: list[CandidateAssessment],
    rejected: list[CandidateAssessment],
    lead_summary: str,
    outreach_message: str,
    identity_decision: str,
    report_markdown: str,
    trace_out_path: str | None,
) -> dict[str, Any]:
    discovery_notes = build_discovery_notes(context.trace, accepted + manual_review + rejected)
    merged = merge_assessment_evidence(profile, accepted, manual_review)
    return {
        "execution_mode": context.actual_mode,
        "model_info": {
            "requested_mode": context.requested_mode,
            "requested_model": context.requested_model,
            "active_model": context.active_model,
            "groq_available": context.groq_available,
            "fallback_reason": context.fallback_reason,
        },
        "lead_profile": asdict(profile),
        "accepted_matches": [asdict(item) for item in accepted[:MAX_REPORT_MATCHES]],
        "manual_review_candidates": [asdict(item) for item in manual_review[:MAX_REPORT_MATCHES]],
        "rejected_candidates": [asdict(item) for item in rejected[:MAX_REPORT_MATCHES]],
        "merged_profile": merged,
        "identity_decision": identity_decision,
        "discovery_notes": discovery_notes,
        "lead_summary": lead_summary,
        "outreach_message": outreach_message,
        "agent_trace": context.trace if not trace_out_path else None,
        "agent_trace_path": trace_out_path or "",
        "report_markdown": report_markdown,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a free, public-web lead enrichment report from a LinkedIn profile URL."
    )
    parser.add_argument("linkedin_url", nargs="?", help="Public LinkedIn profile URL")
    parser.add_argument("--markdown-out", help="Where to save the Markdown report")
    parser.add_argument("--json-out", help="Where to save the JSON payload")
    parser.add_argument(
        "--mode",
        choices=["auto", "agent", "fallback"],
        default=DEFAULT_EXECUTION_MODE,
        help="Execution mode: use Groq agent flow when available, or force fallback heuristics.",
    )
    parser.add_argument("--model", help="Preferred Groq model, for example llama-3.1-8b-instant")
    parser.add_argument("--trace-out", help="Where to save the agent trace JSON")
    args = parser.parse_args()
    linkedin_url, markdown_out, json_out = resolve_cli_inputs(args)

    session = get_session()
    context = resolve_execution_context(session, args.mode, args.model)
    add_trace(
        context.trace,
        "execution_mode",
        "Resolved execution mode.",
        {
            "requested_mode": context.requested_mode,
            "actual_mode": context.actual_mode,
            "requested_model": context.requested_model,
            "active_model": context.active_model,
            "groq_available": context.groq_available,
            "fallback_reason": context.fallback_reason,
        },
    )
    profile = extract_linkedin_profile(session, linkedin_url)
    accepted, manual_review, rejected, lead_summary, outreach_message, identity_decision = run_enrichment_pipeline(
        profile, session, context
    )
    report_markdown = render_markdown_report(
        profile,
        context,
        accepted,
        manual_review,
        rejected,
        lead_summary,
        outreach_message,
        identity_decision,
    )

    trace_out_path = args.trace_out or ""
    if trace_out_path:
        with open(trace_out_path, "w", encoding="utf-8") as trace_file:
            json.dump(context.trace, trace_file, indent=2, ensure_ascii=False)

    payload = build_output_payload(
        profile,
        context,
        accepted,
        manual_review,
        rejected,
        lead_summary,
        outreach_message,
        identity_decision,
        report_markdown,
        trace_out_path or None,
    )

    with open(markdown_out, "w", encoding="utf-8") as markdown_file:
        markdown_file.write(report_markdown)

    with open(json_out, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, ensure_ascii=False)

    print(
        textwrap.dedent(
            f"""
            Report written to:
              Markdown: {markdown_out}
              JSON: {json_out}
              Trace: {trace_out_path or 'embedded in JSON output'}

            Accepted matches:
            """
        ).strip()
    )
    if not accepted:
        print("- No accepted external profile matches were verified.")
    else:
        for assessment in accepted[:5]:
            print(
                f"- {assessment.match.platform}: {assessment.match.candidate_url} "
                f"[{assessment.match.confidence_label} {assessment.match.confidence_score}/100]"
            )


if __name__ == "__main__":
    main()
