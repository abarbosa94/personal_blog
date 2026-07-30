from __future__ import annotations

from dataclasses import replace
from difflib import SequenceMatcher
from html import unescape
from io import BytesIO
import re
from typing import Any
from urllib.parse import quote
from xml.etree import ElementTree

from .http import HttpClient
from .models import (
    Affiliation,
    Paper,
    ReconciledPaper,
    ReconciliationDiagnostic,
)
from .parsing import normalize_title


class OpenAlexReconciler:
    def __init__(
        self,
        http: HttpClient,
        api_key: str | None = None,
        *,
        use_openalex: bool = True,
        use_affiliation_fallback: bool = True,
    ) -> None:
        self.http = http
        self.api_key = api_key
        self.use_openalex = use_openalex
        self.use_affiliation_fallback = use_affiliation_fallback

    def reconcile(self, paper: Paper) -> ReconciledPaper:
        diagnostics: list[ReconciliationDiagnostic] = []
        if self.use_affiliation_fallback:
            paper = self._with_discovered_authors(paper, diagnostics)
        if not self.use_openalex:
            paper = self._with_discovered_pdf(paper, diagnostics)
            diagnostics.append(
                ReconciliationDiagnostic("openalex", "skipped", "PDF-only rerun")
            )
            affiliations = self._pdf_affiliations(paper, diagnostics)
            return ReconciledPaper(
                paper,
                None,
                "pdf+ror" if affiliations else None,
                affiliations,
                tuple(diagnostics),
            )
        if paper.doi:
            work = self._get_by_doi(paper.doi, diagnostics)
            if work:
                return self._from_work_with_fallback(
                    paper, work, "doi", diagnostics
                )
        work = self._get_by_title(paper.title, diagnostics)
        if work and self.title_similarity(
            paper.title, work.get("title") or ""
        ) >= 0.95:
            return self._from_work_with_fallback(
                paper, work, "title", diagnostics
            )
        if work:
            diagnostics.append(
                ReconciliationDiagnostic(
                    "openalex_title",
                    "ambiguous_match",
                    "Best candidate was below the 0.95 title-similarity threshold",
                )
            )
        if not self.use_affiliation_fallback:
            diagnostics.append(
                ReconciliationDiagnostic(
                    "affiliation_fallback", "skipped", "Fast census pass"
                )
            )
            affiliations = ()
        else:
            paper = self._with_discovered_pdf(paper, diagnostics)
            affiliations = self._pdf_affiliations(paper, diagnostics)
        return ReconciledPaper(
            paper,
            None,
            "pdf+ror" if affiliations else None,
            affiliations,
            tuple(diagnostics),
        )

    def _with_discovered_pdf(
        self,
        paper: Paper,
        diagnostics: list[ReconciliationDiagnostic],
    ) -> Paper:
        """Discover an AAAI PDF from its stable article page when Crossref omits it."""

        if paper.pdf_url or paper.venue_key != "aaai" or not paper.doi:
            return paper
        article_id = paper.doi.rsplit(".", 1)[-1]
        if not article_id.isdigit():
            return paper
        article_url = (
            f"https://ojs.aaai.org/index.php/AAAI/article/view/{article_id}"
        )
        try:
            html = self.http.get_text(article_url)
        except Exception as error:
            diagnostics.append(self._error("aaai_pdf_discovery", error))
            return paper
        match = re.search(
            r"""(?:href|content)=["']([^"']+/article/(?:download|view)/"""
            + re.escape(article_id)
            + r"""/\d+[^"']*)["']""",
            html,
            re.IGNORECASE,
        )
        if not match:
            diagnostics.append(
                ReconciliationDiagnostic(
                    "aaai_pdf_discovery", "not_found", article_url
                )
            )
            return paper
        from urllib.parse import urljoin

        pdf_url = urljoin(article_url, match.group(1))
        diagnostics.append(
            ReconciliationDiagnostic(
                "aaai_pdf_discovery", "success", pdf_url
            )
        )
        return replace(paper, official_url=article_url, pdf_url=pdf_url)

    def _with_discovered_authors(
        self,
        paper: Paper,
        diagnostics: list[ReconciliationDiagnostic],
    ) -> Paper:
        """Recover authors from official citation metadata when enumeration omitted them."""

        if paper.authors or not paper.official_url:
            return paper
        try:
            html = self.http.get_text(paper.official_url)
        except Exception as error:
            diagnostics.append(self._error("author_discovery", error))
            return paper
        authors: list[str] = []
        for tag in re.findall(r"<meta\b[^>]*>", html, re.IGNORECASE):
            attributes: dict[str, str] = {}
            for match in re.finditer(
                r"""([:\w-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""",
                tag,
                re.DOTALL,
            ):
                value = next(
                    group for group in match.groups()[1:] if group is not None
                )
                attributes[match.group(1).casefold()] = unescape(value).strip()
            if attributes.get("name", "").casefold() != "citation_author":
                continue
            author = attributes.get("content", "")
            if author and author not in authors:
                authors.append(author)
        diagnostics.append(
            ReconciliationDiagnostic(
                "author_discovery",
                "success" if authors else "not_found",
                f"{len(authors)} authors",
            )
        )
        return replace(paper, authors=tuple(authors)) if authors else paper

    def _from_work_with_fallback(
        self,
        paper: Paper,
        work: dict[str, Any],
        match_method: str,
        diagnostics: list[ReconciliationDiagnostic] | None = None,
    ) -> ReconciledPaper:
        diagnostics = diagnostics if diagnostics is not None else []
        reconciled = self.from_work(paper, work, match_method)
        if reconciled.affiliations:
            return ReconciledPaper(
                reconciled.paper,
                reconciled.openalex_id,
                reconciled.match_method,
                reconciled.affiliations,
                tuple(diagnostics),
            )
        if not self.use_affiliation_fallback:
            diagnostics.append(
                ReconciliationDiagnostic(
                    "affiliation_fallback", "skipped", "Fast census pass"
                )
            )
            return ReconciledPaper(
                reconciled.paper,
                reconciled.openalex_id,
                reconciled.match_method,
                (),
                tuple(diagnostics),
            )
        affiliations: tuple[Affiliation, ...] = ()
        grobid_url = (work.get("content_urls") or {}).get("grobid_xml")
        if grobid_url and self.api_key:
            try:
                xml_text = self.http.get_text(self._with_key(grobid_url))
                affiliations = GrobidAffiliationExtractor.parse(xml_text)
                diagnostics.append(
                    ReconciliationDiagnostic(
                        "grobid", "success", f"{len(affiliations)} affiliations"
                    )
                )
            except Exception as error:
                diagnostics.append(self._error("grobid", error))
                affiliations = ()
        if not affiliations and paper.pdf_url:
            affiliations = self._pdf_affiliations(paper, diagnostics)
        if not affiliations and not paper.pdf_url:
            paper = self._with_discovered_pdf(paper, diagnostics)
            if paper.pdf_url:
                affiliations = self._pdf_affiliations(paper, diagnostics)
        if not affiliations:
            return ReconciledPaper(
                reconciled.paper,
                reconciled.openalex_id,
                reconciled.match_method,
                (),
                tuple(diagnostics),
            )
        return ReconciledPaper(
            paper=paper,
            openalex_id=reconciled.openalex_id,
            match_method=f"{match_method}+affiliation-fallback",
            affiliations=affiliations,
            diagnostics=tuple(diagnostics),
        )

    def _pdf_affiliations(
        self,
        paper: Paper,
        diagnostics: list[ReconciliationDiagnostic] | None = None,
    ) -> tuple[Affiliation, ...]:
        diagnostics = diagnostics if diagnostics is not None else []
        if not paper.pdf_url:
            diagnostics.append(
                ReconciliationDiagnostic("pdf_download", "not_available")
            )
            return ()
        try:
            pdf_bytes = self.http.get_bytes(paper.pdf_url)
            diagnostics.append(
                ReconciliationDiagnostic(
                    "pdf_download", "success", f"{len(pdf_bytes)} bytes"
                )
            )
            raw_values = PdfAffiliationExtractor.extract(pdf_bytes)
            diagnostics.append(
                ReconciliationDiagnostic(
                    "pdf_parse",
                    "success" if raw_values else "no_affiliation_detected",
                    f"{len(raw_values)} candidates",
                )
            )
            affiliations = [
                RorAffiliationResolver(self.http).resolve(value)
                or Affiliation(
                    "",
                    value,
                    CountryMentionExtractor.single_country_code(value),
                    None,
                )
                for value in raw_values
            ]
            # PDF text extraction often breaks a single affiliation across
            # several lines. Preserve directly stated countries even when ROR
            # cannot resolve the resulting organization fragment.
            existing_countries = {
                affiliation.country_code
                for affiliation in affiliations
                if affiliation.country_code
            }
            for country_code in PdfAffiliationExtractor.country_codes(pdf_bytes):
                if country_code in existing_countries:
                    continue
                affiliations.append(
                    Affiliation(
                        "",
                        f"Country explicitly stated in PDF affiliation block ({country_code})",
                        country_code,
                        None,
                    )
                )
            diagnostics.append(
                ReconciliationDiagnostic(
                    "country_resolution",
                    "success"
                    if any(item.country_code for item in affiliations)
                    else "unresolved",
                    ",".join(
                        sorted(
                            {
                                item.country_code
                                for item in affiliations
                                if item.country_code
                            }
                        )
                    ),
                )
            )
            return tuple(affiliations)
        except Exception as error:
            stage = (
                "pdf_download"
                if not any(item.stage == "pdf_download" for item in diagnostics)
                else "pdf_parse_or_ror"
            )
            diagnostics.append(self._error(stage, error))
            return ()

    def _with_key(self, url: str) -> str:
        if not self.api_key:
            return url
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}api_key={quote(self.api_key)}"

    def _get_by_doi(
        self, doi: str, diagnostics: list[ReconciliationDiagnostic]
    ) -> dict[str, Any] | None:
        url = self._with_key(
            f"https://api.openalex.org/works/https://doi.org/{quote(doi, safe='/')}"
        )
        try:
            work = self.http.get_json(url)
            diagnostics.append(
                ReconciliationDiagnostic("openalex_doi", "success")
            )
            return work
        except Exception as error:
            diagnostics.append(self._error("openalex_doi", error))
            return None

    def _get_by_title(
        self, title: str, diagnostics: list[ReconciliationDiagnostic]
    ) -> dict[str, Any] | None:
        url = self._with_key(
            "https://api.openalex.org/works?"
            f"search.exact={quote(title, safe='')}&per-page=5"
        )
        try:
            payload = self.http.get_json(url)
        except Exception as error:
            diagnostics.append(self._error("openalex_title", error))
            return None
        results = payload.get("results", [])
        if not results:
            diagnostics.append(
                ReconciliationDiagnostic("openalex_title", "not_found")
            )
            return None
        diagnostics.append(
            ReconciliationDiagnostic(
                "openalex_title", "success", f"{len(results)} candidates"
            )
        )
        return max(
            results,
            key=lambda work: self.title_similarity(
                title, work.get("title") or ""
            ),
        )

    @staticmethod
    def _error(stage: str, error: Exception) -> ReconciliationDiagnostic:
        from urllib.error import HTTPError

        if isinstance(error, HTTPError) and error.code == 404:
            return ReconciliationDiagnostic(stage, "not_found", "HTTP 404")
        status = f"HTTP {error.code}" if isinstance(error, HTTPError) else type(error).__name__
        return ReconciliationDiagnostic(stage, "api_error", status)

    @staticmethod
    def title_similarity(left: str, right: str) -> float:
        return SequenceMatcher(None, normalize_title(left), normalize_title(right)).ratio()

    @staticmethod
    def from_work(
        paper: Paper, work: dict[str, Any], match_method: str
    ) -> ReconciledPaper:
        unique: dict[str, Affiliation] = {}
        for authorship in work.get("authorships", []):
            for institution in authorship.get("institutions", []):
                institution_id = institution.get("id") or ""
                key = institution_id or (
                    f"{institution.get('display_name', '')}|"
                    f"{institution.get('country_code', '')}"
                )
                if not key:
                    continue
                unique[key] = Affiliation(
                    institution_id=institution_id,
                    institution_name=institution.get("display_name") or "",
                    country_code=institution.get("country_code"),
                    institution_type=institution.get("type"),
                )
        return ReconciledPaper(
            paper=paper,
            openalex_id=work.get("id"),
            match_method=match_method,
            affiliations=tuple(unique.values()),
        )


def full_country_weights(paper: ReconciledPaper) -> dict[str, float]:
    return {country: 1.0 for country in paper.countries}


def fractional_country_weights(paper: ReconciledPaper) -> dict[str, float]:
    countries = paper.countries
    if not countries:
        return {}
    weight = 1.0 / len(countries)
    return {country: weight for country in countries}


class GrobidAffiliationExtractor:
    """Extract organization and country labels from TEI affiliation elements."""

    @staticmethod
    def parse(xml_text: str) -> tuple[Affiliation, ...]:
        root = ElementTree.fromstring(xml_text)
        values: dict[str, Affiliation] = {}
        for element in root.iter():
            if GrobidAffiliationExtractor._local_name(element.tag) != "affiliation":
                continue
            organizations = [
                " ".join(child.itertext()).strip()
                for child in element.iter()
                if GrobidAffiliationExtractor._local_name(child.tag) == "orgName"
            ]
            organization = "; ".join(dict.fromkeys(filter(None, organizations)))
            country_code: str | None = None
            for child in element.iter():
                if GrobidAffiliationExtractor._local_name(child.tag) == "country":
                    country_code = (
                        child.attrib.get("key")
                        or child.attrib.get("{http://www.w3.org/XML/1998/namespace}id")
                    )
                    if country_code:
                        country_code = country_code.upper()
                    break
            if not organization:
                continue
            key = f"{organization}|{country_code or ''}"
            values[key] = Affiliation(
                institution_id="",
                institution_name=organization,
                country_code=country_code,
                institution_type=None,
            )
        return tuple(values.values())

    @staticmethod
    def _local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]


class PdfAffiliationExtractor:
    """Extract conservative affiliation candidates from the first PDF pages."""

    NON_AUTHOR_ORGANIZATIONS = {
        "association for computational linguistics",
    }
    KEYWORDS = re.compile(
        r"\b("
        r"university|universidade|universidad|universit[aäé]t|université|"
        r"department|institute|institut|laborator(?:y|ies)|college|school|"
        r"research\s+(?:cent(?:er|re)|institut(?:e|ion)|laborator(?:y|ies))|"
        r"google|deepmind|microsoft|meta|amazon|nvidia|openai|anthropic|"
        r"corporation|company|inc\.?|ltd\.?|technology|technologies|research"
        r")\b",
        re.IGNORECASE,
    )

    @staticmethod
    def extract(pdf_bytes: bytes) -> tuple[str, ...]:
        text = PdfAffiliationExtractor.extract_text(pdf_bytes)
        return PdfAffiliationExtractor.candidate_lines(text)

    @staticmethod
    def candidate_lines(text: str) -> tuple[str, ...]:
        front_matter = PdfAffiliationExtractor.affiliation_region(text)
        values: list[str] = []
        # Split compact ``1Org2Org`` lines before considering the original
        # line. This preserves affiliations that share one extracted PDF line.
        for line in front_matter.splitlines():
            normalized_line = " ".join(line.split())
            markers = list(
                re.finditer(r"(?<!\d)(\d+)(?=\s*[A-Z])", normalized_line)
            )
            for index, marker in enumerate(markers):
                end = (
                    markers[index + 1].start()
                    if index + 1 < len(markers)
                    else len(normalized_line)
                )
                cleaned_piece = normalized_line[marker.end() : end].strip(
                    " ,;*0123456789"
                )
                if not 4 <= len(cleaned_piece) <= 300:
                    continue
                if PdfAffiliationExtractor._is_non_author_organization(
                    cleaned_piece
                ):
                    continue
                if (
                    PdfAffiliationExtractor.KEYWORDS.search(cleaned_piece)
                    or RorAffiliationResolver.known_affiliation(cleaned_piece)
                ):
                    values.append(cleaned_piece)
        for line in front_matter.splitlines():
            cleaned = " ".join(line.split()).strip(" ,;*†‡0123456789")
            if not 4 <= len(cleaned) <= 300:
                continue
            if PdfAffiliationExtractor._is_non_author_organization(cleaned):
                continue
            if PdfAffiliationExtractor.KEYWORDS.search(cleaned):
                values.append(cleaned)
        # Some papers print a single unnumbered company affiliation directly
        # below the authors. Column-ordered PDF text can place the correspondence
        # footnote much later, outside ``affiliation_region``. Inspect only the
        # pre-abstract front matter as a conservative fallback.
        if not values:
            pre_abstract = re.split(
                r"(?im)^\s*abstract\s*$", text, maxsplit=1
            )[0]
            for line in pre_abstract.splitlines():
                cleaned = " ".join(line.split()).strip(
                    " ,;*â€ â€¡0123456789"
                )
                if not 4 <= len(cleaned) <= 180:
                    continue
                if RorAffiliationResolver.known_affiliation(cleaned):
                    values.append(cleaned)
        return tuple(dict.fromkeys(values))

    @staticmethod
    def _is_non_author_organization(value: str) -> bool:
        normalized = RorAffiliationResolver._normalized(value)
        return any(
            excluded in normalized
            for excluded in PdfAffiliationExtractor.NON_AUTHOR_ORGANIZATIONS
        )

    @staticmethod
    def extract_text(pdf_bytes: bytes) -> str:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(pdf_bytes))
        return "\n".join(
            page.extract_text() or "" for page in reader.pages[: min(2, len(reader.pages))]
        )

    @staticmethod
    def country_codes(pdf_bytes: bytes) -> tuple[str, ...]:
        return CountryMentionExtractor.country_codes(
            PdfAffiliationExtractor.affiliation_region(
                PdfAffiliationExtractor.extract_text(pdf_bytes)
            )
        )

    @staticmethod
    def affiliation_region(text: str) -> str:
        """Return the affiliation block immediately preceding correspondence.

        PDF column ordering often places the affiliation footnote after the
        ``Introduction`` heading, so section-based truncation is unreliable.
        PMLR places affiliations directly before ``Correspondence to``. Locate a
        numbered organization line within the preceding window and keep that
        suffix. The conference footer follows correspondence and is excluded.
        """

        normalized = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
        parts = re.split(
            r"(?i)\bcorrespondence\s+to\s*:", normalized, maxsplit=1
        )
        if len(parts) == 1:
            # In ordinary AAAI/ACL layouts affiliations precede the abstract.
            # Stopping there prevents countries mentioned in the abstract or
            # introduction (for example experimental languages) from being
            # mistaken for author locations.
            return re.split(
                r"(?im)^\s*(?:abstract|(?:\d+\.?\s+)?introduction(?:\s+and\s+motivation)?)\s*$",
                normalized,
                maxsplit=1,
            )[0]
        before_correspondence = parts[0]
        lines = before_correspondence.splitlines()
        window_start = max(0, len(lines) - 40)
        start: int | None = None
        marker = re.compile(r"(?:^|\s)[*#†‡,]*\s*\d+\s*\w")
        for index in range(window_start, len(lines)):
            line = lines[index]
            keyword_text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", line)
            if marker.search(line) and PdfAffiliationExtractor.KEYWORDS.search(
                keyword_text
            ):
                start = index
                break
        if start is None:
            start = max(0, len(lines) - 15)
        return "\n".join(lines[start:])


class CountryMentionExtractor:
    """Map explicit country mentions in PDF affiliation text to ISO alpha-2."""

    COUNTRY_ALIASES = {
        "AE": ("united arab emirates", "uae"),
        "AT": ("austria", "austrian", "linz"),
        "AU": ("australia", "australian"),
        "BR": ("brazil",),
        "CA": ("canada",),
        "CH": ("switzerland",),
        "CN": ("china", "chinese", "p.r. china", "pr china"),
        "CO": ("colombia",),
        "DE": ("germany", "german"),
        "DK": ("denmark", "danish"),
        "EC": ("ecuador",),
        "ES": ("spain", "spanish"),
        "FR": ("france", "french"),
        "GB": ("united kingdom", "u.k.", "uk", "england"),
        "HK": ("hong kong",),
        "HU": ("hungary", "hungarian"),
        "ID": ("indonesia", "indonesian"),
        "IE": ("ireland",),
        "IL": ("israel", "israeli"),
        "IN": ("india", "indian"),
        "IT": ("italy", "italian"),
        "JP": ("japan", "japanese"),
        "KE": ("kenya", "kenyan"),
        "KR": ("south korea", "south korean", "republic of korea"),
        "MA": ("morocco", "moroccan"),
        "MK": ("north macedonia", "macedonia"),
        "MO": ("macao", "macau"),
        "NG": ("nigeria", "nigerian"),
        "NL": ("netherlands", "dutch"),
        "PT": ("portugal", "portuguese"),
        "RO": ("romania", "romanian"),
        "RU": ("russia", "russian", "russian federation"),
        "RW": ("rwanda", "rwandan"),
        "SA": ("saudi arabia", "saudi"),
        "SE": ("sweden", "swedish"),
        "SG": ("singapore",),
        "ZA": ("south africa", "south african"),
        "US": (
            "united states of america",
            "united states",
            "u.s.a.",
            "usa",
            "new york",
            "san francisco",
            "palo alto",
            "menlo park",
            "mountain view",
        ),
    }
    _PATTERNS = {
        code: tuple(
            re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)", re.IGNORECASE)
            for alias in aliases
        )
        for code, aliases in COUNTRY_ALIASES.items()
    }

    @classmethod
    def country_codes(cls, text: str) -> tuple[str, ...]:
        normalized = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
        normalized = re.sub(r"\s+", " ", normalized)
        # The Shenzhen campus is in mainland China; "Hong Kong" is part of
        # the institution's name rather than a location statement.
        normalized = re.sub(
            r"Chinese University of Hong Kong,?\s+Shenzhen",
            "CUHK Shenzhen",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"(?:The\s+)?Hong Kong University of Science and Technology\s*"
            r"\(\s*Guangzhou\s*\)",
            "HKUST Guangzhou",
            normalized,
            flags=re.IGNORECASE,
        )
        values = {
            code
            for code, patterns in cls._PATTERNS.items()
            if any(pattern.search(normalized) for pattern in patterns)
        }
        # ``US`` and ``UK`` are commonly printed as countries. Do not generalize
        # this to every ISO code: ``CA`` and similar values are also state or
        # province abbreviations in affiliation addresses.
        if re.search(r"(?<![A-Z])US(?![A-Z])", normalized):
            values.add("US")
        if re.search(r"(?<![A-Z])UK(?![A-Z])", normalized):
            values.add("GB")
        return tuple(sorted(values))

    @classmethod
    def single_country_code(cls, text: str) -> str | None:
        codes = cls.country_codes(text)
        return codes[0] if len(codes) == 1 else None


class RorAffiliationResolver:
    """Resolve a raw affiliation only when the ROR service marks it chosen."""

    endpoint = "https://api.ror.org/v2/organizations"
    # Reviewed, unambiguous organization aliases. These are deliberately narrow:
    # matching requires the complete normalized phrase, not an individual token.
    KNOWN_ORGANIZATIONS = {
        "airi": (
            "Artificial Intelligence Research Institute",
            "RU",
            "facility",
        ),
        "cerai iit madras": (
            "Centre for Responsible AI, IIT Madras",
            "IN",
            "education",
        ),
        "epfl": ("EPFL", "CH", "education"),
        "indian institute of science": (
            "Indian Institute of Science",
            "IN",
            "education",
        ),
        "iiit hyderabad": (
            "International Institute of Information Technology, Hyderabad",
            "IN",
            "education",
        ),
        "hkust": (
            "Hong Kong University of Science and Technology",
            "HK",
            "education",
        ),
        "johns hopkins bloomberg school of public health": (
            "Johns Hopkins Bloomberg School of Public Health",
            "US",
            "education",
        ),
        "new york university abu dhabi": (
            "New York University Abu Dhabi",
            "AE",
            "education",
        ),
        "sensetime research": ("SenseTime Research", "CN", "company"),
        "kaist ai": ("KAIST", "KR", "education"),
        "liacc feup university of porto": (
            "University of Porto",
            "PT",
            "education",
        ),
        "mbzuai": (
            "Mohamed bin Zayed University of Artificial Intelligence",
            "AE",
            "education",
        ),
        "microsoft research asia": ("Microsoft Research Asia", "CN", "company"),
        "microsoft": ("Microsoft", "US", "company"),
        "nec laboratories europe": (
            "NEC Laboratories Europe",
            "DE",
            "company",
        ),
        "northeastern university": (
            "Northeastern University",
            "US",
            "education",
        ),
        "penn state university": (
            "Pennsylvania State University",
            "US",
            "education",
        ),
        "university of melbourne": (
            "University of Melbourne",
            "AU",
            "education",
        ),
        "university of sydney": (
            "University of Sydney",
            "AU",
            "education",
        ),
        "sungkyunkwan university": (
            "Sungkyunkwan University",
            "KR",
            "education",
        ),
        "huawei noah s ark lab": ("Huawei Noah's Ark Lab", "CN", "company"),
        "guangzhou quwan network technology": (
            "Guangzhou Quwan Network Technology",
            "CN",
            "company",
        ),
        "origin research": ("Origin Research", "US", "company"),
        "origin wireless": ("Origin Wireless", "US", "company"),
        "salesforce": ("Salesforce", "US", "company"),
        "skoltech": (
            "Skolkovo Institute of Science and Technology",
            "RU",
            "education",
        ),
        "universitas indonesia": (
            "Universitas Indonesia",
            "ID",
            "education",
        ),
        "wsai iit madras": (
            "Wadhwani School of Data Science and AI, IIT Madras",
            "IN",
            "education",
        ),
    }
    GENERIC_TOKENS = {
        "and",
        "center",
        "centre",
        "college",
        "company",
        "corporation",
        "department",
        "for",
        "institute",
        "laboratory",
        "lab",
        "of",
        "public",
        "research",
        "school",
        "science",
        "technology",
        "health",
        "the",
        "university",
    }
    AMBIGUOUS_TRUNCATED_NAMES = {
        "institute of biomedical engineering",
        "institute for machine learning",
    }
    # These labels identify units shared by multiple institutions. A bare unit
    # name must not inherit the country of whichever ROR result ranks first.
    AMBIGUOUS_BARE_NAMES = {
        "centre for artificial intelligence and robotics",
        "google deepmind",
        "institute for machine learning",
    }
    # Branch-level evidence is keyed by both author and organization. This
    # avoids assigning a multinational's headquarters country to every paper.
    KNOWN_AUTHOR_BRANCHES = {
        ("simon see", "nvidia ai technology center"): (
            "NVIDIA AI Technology Center, Singapore",
            "SG",
            "company",
        ),
        ("zhuo chen", "bytedance seed"): (
            "ByteDance Seed, United States",
            "US",
            "company",
        ),
        ("xiaomo liu", "jpmorgan ai research"): (
            "JPMorgan AI Research, New York",
            "US",
            "company",
        ),
        ("mingbao lin", "skywork ai"): (
            "Skywork AI, Singapore",
            "SG",
            "company",
        ),
        ("nicola cancedda", "fair at meta"): (
            "FAIR at Meta, London",
            "GB",
            "company",
        ),
        ("alan schelten", "genai at meta"): (
            "GenAI at Meta, London",
            "GB",
            "company",
        ),
        ("tara fowler", "genai at meta"): (
            "GenAI at Meta, New York",
            "US",
            "company",
        ),
        ("yuxi wang", "centre for artificial intelligence and robotics"): (
            "Centre for Artificial Intelligence and Robotics, Hong Kong",
            "HK",
            "facility",
        ),
        ("jun hou", "centre for artificial intelligence and robotics"): (
            "Centre for Artificial Intelligence and Robotics, Hong Kong",
            "HK",
            "facility",
        ),
        ("zhaoxiang zhang", "centre for artificial intelligence and robotics"): (
            "Centre for Artificial Intelligence and Robotics, Hong Kong",
            "HK",
            "facility",
        ),
    }
    # Some venue enumerators do not expose authors. Keep narrowly evidenced
    # paper-and-organization branch facts so those records can still retain a
    # branch country without generalizing a multinational's headquarters.
    KNOWN_PAPER_BRANCHES = {
        (
            "advancing zero shot text to speech intelligibility across diverse "
            "domains via preference alignment",
            "bytedance seed",
        ): ("ByteDance Seed, United States", "US"),
        (
            "cocolex confidence guided copy based decoding for grounded legal "
            "text generation",
            "jpmorgan ai research",
        ): ("JPMorgan AI Research, New York", "US"),
    }
    # Last-resort evidence for a reviewed paper whose PDF text extraction loses
    # the Meta affiliation line. Requiring both exact title and author keeps
    # these branch facts from following an author to unrelated future papers.
    KNOWN_AUTHOR_PAPER_BRANCHES = {
        ("nicola cancedda", "hallulens llm hallucination benchmark"): (
            "FAIR at Meta, London",
            "GB",
        ),
        ("alan schelten", "hallulens llm hallucination benchmark"): (
            "GenAI at Meta, London",
            "GB",
        ),
        ("tara fowler", "hallulens llm hallucination benchmark"): (
            "GenAI at Meta, New York",
            "US",
        ),
    }

    def __init__(self, http: HttpClient) -> None:
        self.http = http

    @staticmethod
    def _normalized(value: str) -> str:
        import unicodedata

        value = unicodedata.normalize("NFKD", value)
        value = "".join(char for char in value if not unicodedata.combining(char))
        return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()

    @classmethod
    def known_affiliation(cls, raw_affiliation: str) -> Affiliation | None:
        normalized = cls._normalized(raw_affiliation)
        padded = f" {normalized} "
        # Prefer longer aliases so Microsoft Research Asia is not reduced to
        # the parent Microsoft organization.
        for alias in sorted(cls.KNOWN_ORGANIZATIONS, key=len, reverse=True):
            normalized_alias = cls._normalized(alias)
            if f" {normalized_alias} " not in padded:
                continue
            name, country, institution_type = cls.KNOWN_ORGANIZATIONS[alias]
            return Affiliation("", name, country, institution_type)
        return None

    @classmethod
    def _lexically_supported(cls, raw: str, display_name: str) -> bool:
        raw_normalized = cls._normalized(raw)
        display_normalized = cls._normalized(display_name)
        if not raw_normalized or not display_normalized:
            return False
        if f" {display_normalized} " in f" {raw_normalized} ":
            return True
        raw_tokens = set(raw_normalized.split()) - cls.GENERIC_TOKENS
        display_tokens = set(display_normalized.split()) - cls.GENERIC_TOKENS
        return len(raw_tokens & display_tokens) >= 2

    @classmethod
    def author_branch_affiliations(
        cls,
        authors: tuple[str, ...],
        affiliations: tuple[Affiliation, ...] | list[Affiliation],
        organization_candidates: tuple[str, ...] = (),
        paper_title: str = "",
    ) -> tuple[Affiliation, ...]:
        normalized_authors = {cls._normalized(author) for author in authors}
        organization_text = " ".join(
            cls._normalized(item.institution_name) for item in affiliations
        )
        organization_text += " " + " ".join(
            cls._normalized(candidate) for candidate in organization_candidates
        )
        values = []
        for (author, organization), (name, country, institution_type) in (
            cls.KNOWN_AUTHOR_BRANCHES.items()
        ):
            if author not in normalized_authors:
                continue
            if organization not in organization_text:
                continue
            values.append(Affiliation("", name, country, institution_type))
        normalized_title = cls._normalized(paper_title)
        for (author, title), (name, country) in (
            cls.KNOWN_AUTHOR_PAPER_BRANCHES.items()
        ):
            if author not in normalized_authors or title != normalized_title:
                continue
            values.append(Affiliation("", name, country, "company"))
        for (title, organization), (name, country) in cls.KNOWN_PAPER_BRANCHES.items():
            if title != normalized_title:
                continue
            if organization not in organization_text:
                continue
            values.append(Affiliation("", name, country, "company"))
        return tuple(values)

    @classmethod
    def pdf_supports_affiliation(
        cls, affiliation: Affiliation, affiliation_region: str
    ) -> bool:
        """Require distinctive PDF support before retaining an OpenAlex institution."""

        display = cls._normalized(affiliation.institution_name)
        region = cls._normalized(affiliation_region)
        if not display or not region:
            return False
        if display in cls.AMBIGUOUS_BARE_NAMES:
            return False
        display_without_article = re.sub(r"^the\s+", "", display)
        if f" {display_without_article} " in f" {region} ":
            return True
        display_tokens = set(display.split()) - cls.GENERIC_TOKENS
        region_tokens = set(region.split()) - cls.GENERIC_TOKENS
        if not display_tokens:
            return False
        overlap = display_tokens & region_tokens
        return len(overlap) >= 2 and len(overlap) / len(display_tokens) >= 0.6

    def resolve(self, raw_affiliation: str) -> Affiliation | None:
        known = self.known_affiliation(raw_affiliation)
        if known:
            return known
        if self._normalized(raw_affiliation) in self.AMBIGUOUS_BARE_NAMES:
            return None
        url = f"{self.endpoint}?affiliation={quote(raw_affiliation, safe='')}"
        payload = self.http.get_json(url)
        for item in payload.get("items", []):
            if not item.get("chosen"):
                continue
            organization = item.get("organization") or {}
            affiliation = self._from_organization(raw_affiliation, organization)
            if affiliation:
                return affiliation
        # The affiliation endpoint can decline short but exact official aliases
        # such as "UC Merced". Query all organization names as a conservative
        # fallback, accepting only an exact normalized official name or alias.
        query_url = f"{self.endpoint}?query={quote(raw_affiliation, safe='')}"
        query_payload = self.http.get_json(query_url)
        normalized_raw = self._normalized(raw_affiliation)
        # Bare acronyms are often shared across institutions (for example,
        # UCLA and MA). Require a multi-token alias and reject aliases that
        # identify more than one ROR organization.
        if len(normalized_raw.split()) < 2:
            return None
        exact_organizations: dict[str, dict[str, Any]] = {}
        for item in query_payload.get("items", []):
            organization = item.get("organization") or item
            names = organization.get("names") or []
            normalized_names = {
                self._normalized(name.get("value") or "") for name in names
            }
            if normalized_raw not in normalized_names:
                continue
            key = organization.get("id") or json.dumps(
                organization, sort_keys=True
            )
            exact_organizations[key] = organization
        if len(exact_organizations) != 1:
            return None
        organization = next(iter(exact_organizations.values()))
        affiliation = self._from_organization(raw_affiliation, organization)
        if affiliation:
            return affiliation
        return None

    @classmethod
    def _from_organization(
        cls, raw_affiliation: str, organization: dict[str, Any]
    ) -> Affiliation | None:
        names = organization.get("names") or []
        display_name = next(
            (
                name.get("value")
                for name in names
                if "ror_display" in (name.get("types") or [])
            ),
            raw_affiliation,
        )
        if (
            cls._normalized(display_name) in cls.AMBIGUOUS_TRUNCATED_NAMES
            and cls._normalized(raw_affiliation) != cls._normalized(display_name)
        ):
            return None
        if not cls._lexically_supported(raw_affiliation, display_name):
            normalized_names = {
                cls._normalized(name.get("value") or "") for name in names
            }
            if cls._normalized(raw_affiliation) not in normalized_names:
                return None
        locations = organization.get("locations") or []
        country_code = None
        if locations:
            country_code = (
                locations[0].get("geonames_details") or {}
            ).get("country_code")
        types = organization.get("types") or []
        return Affiliation(
            institution_id=organization.get("id") or "",
            institution_name=display_name,
            country_code=country_code,
            institution_type=types[0] if types else None,
        )
