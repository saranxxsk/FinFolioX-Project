"""
ENHANCED MCP DATA SERVER HOLD FinFolioX v2.2 (Future Events Added)
================================================================
Model Context Protocol (MCP) Server with Real-Time Macro Intelligence.

WHAT'S NEW IN v2.2:
  NEW TIER 8 (weight 0.75) HOLD Future Event Scout
    Fetches upcoming events that could move the stock in the next 30 days:
      • Earnings date + analyst expectations (Yahoo Finance calendar API)
      • FOMC meeting dates with expected rate decision context
      • Upcoming product launches / conferences (Google News forward search)
      • Scheduled SEC filings (next 10-Q/8-K window)
      • Reddit upcoming event mentions ("earnings play", "catalyst", "next week")
    These are returned as structured text for the LLM future scorer in
    sentiment_agent.py to reason about.

ALL PREVIOUS FIXES (v2.1) PRESERVED EXACTLY HOLD nothing changed below Tier 7.

FIXES IN v2.1:
  FIX-1: FRED RSS fallback order improved.
  FIX-2: GDELT conflict-monitor query narrowed to ticker-specific topics.
  FIX-3: EconCalendar investing.com RSS 403 -> BLS/BEA primary.
  FIX-4: VIX numeric fetch (no RSS) -> Yahoo Finance quote API.
  FIX-5: Fuzzy deduplication (first 60 chars).
  FIX-6: Per-source item caps.
"""

import requests
import xml.etree.ElementTree as ET
import re
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

logger = logging.getLogger("MCPServer")


# ==============================================================================
# CONFIGURATION HOLD Tune weights here
# ==============================================================================
TIER_WEIGHTS = {
    "FRED":          1.50,   # Tier 0 HOLD Central Bank / Fed
    "SEC EDGAR":     1.00,   # Tier 1 HOLD Regulatory filings
    "Yahoo Finance": 0.90,   # Tier 2 HOLD Financial news
    "GDELT":         0.80,   # Tier 3 HOLD Geopolitical events
    "EconCalendar":  0.70,   # Tier 4 HOLD Economic data releases
    "MacroFX":       0.60,   # Tier 5 HOLD Commodity / FX macro
    "GoogleTrends":  0.50,   # Tier 6 HOLD Search interest proxy
    "Reddit r/WSB":  0.30,   # Tier 7 HOLD Retail sentiment
    "FutureEvents":  0.75,   # Tier 8 HOLD Upcoming events (NEW v2.2)
}

# Per-source item caps
SOURCE_MAX_ITEMS = {
    "FRED":          4,
    "SEC EDGAR":     3,
    "Yahoo Finance": 4,
    "GDELT":         4,
    "EconCalendar":  3,
    "MacroFX":       3,
    "GoogleTrends":  3,
    "Reddit r/WSB":  3,
    "FutureEvents":  6,   # NEW HOLD up to 6 forward-looking signals
}

# ==============================================================================
# FIX-3: FINANCE RELEVANCE FILTER FOR FUTURE EVENTS
# Prevents garbage (entertainment, sports, politics) from polluting the LLM prompt.
# A future event article must contain at least one of these keywords to be included.
# Two tiers:
#   FINANCE_MUST_KEYWORDS HOLD strong market-moving signals (any one = include)
#   FINANCE_BROAD_KEYWORDS HOLD weaker signals (require 2 matches to include)
# ==============================================================================
FINANCE_MUST_KEYWORDS = [
    # Earnings & guidance
    "earnings", "eps", "revenue", "profit", "guidance", "forecast", "outlook",
    "beat", "miss", "estimate", "quarterly", "annual report", "results",
    # Fed / rates / macro
    "fed", "federal reserve", "fomc", "interest rate", "rate hike", "rate cut",
    "inflation", "cpi", "pce", "gdp", "jobs", "payroll", "unemployment",
    # Company events
    "product launch", "launch", "keynote", "announcement", "conference",
    "merger", "acquisition", "buyback", "dividend", "spinoff", "ipo",
    # Tech / AI / Chips (high-impact for NVDA, AAPL, MSFT, GOOGL)
    "ai", "artificial intelligence", "chip", "semiconductor", "gpu",
    "regulation", "antitrust", "lawsuit", "sec", "fine", "penalty",
    # Macro / FX / commodities
    "opec", "crude oil", "gold", "dollar", "tariff", "trade war",
    "recession", "credit", "bond yield", "treasury bond", "government bond",
    "bond market", "corporate bond", "junk bond", "yield", "debt ceiling",
    # Catalyst keywords
    "catalyst", "breakout", "upgrade", "downgrade", "price target",
    "analyst", "rating", "buy", "sell", "hold",
]

FINANCE_BROAD_KEYWORDS = [
    "stock", "market", "shares", "investors", "wall street", "nasdaq",
    "s&p", "index", "etf", "fund", "portfolio", "trading",
]

# ==============================================================================
# FINANCE BLOCKLIST HOLD explicit non-finance false positives
# "bond" matches financial bonds but also James Bond, Bail Bond, etc.
# Any text containing one of these phrases is immediately excluded,
# even if it also contains a finance keyword.
# ==============================================================================
FINANCE_BLOCKLIST = [
    "james bond",       # Bond film franchise
    "stranger things",  # Netflix
    "stranger thing",
    "james bond",
    "007",
    "taylor swift",
    "super bowl",
    "oscar",
    "emmy",
    "grammy",
    "golden globe",
    "world cup",
    "nba finals",
    "nfl draft",
    "boxing match",
    "premiere",         # film/show premieres
    "celebrity",
    "kardashian",
    "royal family",
]

def _is_finance_relevant(text: str) -> bool:
    """
    Returns True if text passes the finance relevance gate.

    Two-step logic:
      Step 1 HOLD Blocklist check (fast reject):
        If any non-finance phrase is found (e.g. "james bond", "super bowl"),
        return False immediately HOLD even if the text also contains "bond" or "market".
        This prevents false positives like "Who will be the next James Bond?"
        matching the financial keyword "bond".

      Step 2 HOLD Keyword match (accept):
        Strong finance keyword -> accept.
        Two broad finance keywords together -> accept.
    """
    lower = text.lower()

    # Step 1: Hard blocklist HOLD reject non-finance topics immediately
    if any(phrase in lower for phrase in FINANCE_BLOCKLIST):
        return False

    # Step 2: Strong finance keyword -> immediately relevant
    if any(kw in lower for kw in FINANCE_MUST_KEYWORDS):
        return True

    # Step 3: Two broad keywords together -> probably relevant
    broad_hits = sum(1 for kw in FINANCE_BROAD_KEYWORDS if kw in lower)
    return broad_hits >= 2

# GDELT topic filters
GDELT_TOPIC_MAP = {
    "OIL":   ["OPEC crude oil price", "Strait of Hormuz oil", "Iran Saudi Arabia oil"],
    "GOLD":  ["Federal Reserve gold inflation", "dollar gold safe haven"],
    "SPY":   ["Federal Reserve recession S&P 500", "US trade war tariff earnings"],
    "QQQ":   ["AI semiconductor chip", "tech antitrust regulation NVDA"],
    "WTI":   ["OPEC crude oil pipeline", "Iraq Libya oil production"],
    "BTC":   ["Bitcoin crypto regulation SEC stablecoin"],
    "TLT":   ["Federal Reserve interest rate FOMC treasury yield"],
    "GLD":   ["gold inflation hedge dollar central bank reserves"],
    "DEFAULT": ["Federal Reserve recession inflation trade war"],
}

# FOMC 2026 scheduled meeting dates (hardcoded HOLD updated annually)
# Source: federalreserve.gov/monetarypolicy/fomccalendars.htm
FOMC_2026_DATES = [
    ("2026-01-28", "2026-01-29"),
    ("2026-03-18", "2026-03-19"),
    ("2026-04-29", "2026-04-30"),
    ("2026-06-17", "2026-06-18"),
    ("2026-07-29", "2026-07-30"),
    ("2026-09-16", "2026-09-17"),
    ("2026-10-28", "2026-10-29"),
    ("2026-12-09", "2026-12-10"),
]

# FRED series
FRED_SERIES = {
    "FEDFUNDS":   "Federal Funds Rate",
    "CPIAUCSL":   "Consumer Price Index (Inflation)",
    "UNRATE":     "Unemployment Rate",
    "GDP":        "Gross Domestic Product",
    "DGS10":      "10-Year Treasury Yield",
    "DCOILWTICO": "WTI Crude Oil Price (FRED)",
}


class MCPDataServer:
    """
    Enhanced MCP Server v2.2.
    Fetches present news (Tiers 0-7) + upcoming events (Tier 8).
    """

    def __init__(self):
        self.headers = {
            "User-Agent": "FinFolioX_Research_Bot/2.2 (Educational Capstone)"
        }
        self.reddit_headers = {
            "User-Agent": "FinFolioX:v2.2 (by /u/finfolio_admin)"
        }
        self.ticker_to_cik = {}
        self._load_sec_cik_mapping()
        print("   🔌 [MCP Server v2.2] Initialized HOLD 9 tiers active (Tier 8: Future Events)")

    # ==========================================================================
    # UTILITIES
    # ==========================================================================

    def _load_sec_cik_mapping(self):
        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            resp = requests.get(url, headers=self.headers, timeout=5)
            if resp.status_code == 200:
                for entry in resp.json().values():
                    self.ticker_to_cik[entry["ticker"].upper()] = str(entry["cik_str"]).zfill(10)
        except Exception:
            pass

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(re.compile("<.*?>"), "", text)
        text = text.encode("ascii", "ignore").decode("ascii")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _gdelt_topics_for_ticker(self, ticker: str) -> List[str]:
        ticker_upper = ticker.upper()
        for key in GDELT_TOPIC_MAP:
            if key in ticker_upper or ticker_upper in key:
                return GDELT_TOPIC_MAP[key]
        return GDELT_TOPIC_MAP["DEFAULT"]

    # ==========================================================================
    # TIER 0: FRED
    # ==========================================================================

    def fetch_fed_macro(self, ticker: str) -> List[Dict]:
        results = []
        fred_series_to_try = [
            ("FEDFUNDS", "Federal Funds Rate"),
            ("DGS10",    "10-Year Treasury Yield"),
            ("CPIAUCSL", "Consumer Price Index"),
        ]
        for series_id, series_label in fred_series_to_try:
            if len(results) >= SOURCE_MAX_ITEMS["FRED"]:
                break
            try:
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
                resp = requests.get(url, headers=self.headers, timeout=5)
                if resp.status_code == 200:
                    lines = resp.text.strip().split("\n")
                    if len(lines) >= 2:
                        latest = lines[-1].split(",")
                        prev   = lines[-2].split(",") if len(lines) >= 3 else latest
                        if len(latest) == 2:
                            date_val, rate_val = latest[0].strip(), latest[1].strip()
                            prev_val = prev[1].strip() if len(prev) == 2 else rate_val
                            try:
                                change    = float(rate_val) - float(prev_val)
                                direction = "unchanged" if abs(change) < 0.001 else (
                                    "increased" if change > 0 else "decreased")
                                text = (
                                    f"{series_label} as of {date_val}: {rate_val}%. "
                                    f"Rate {direction} from previous period ({prev_val}%)."
                                )
                            except ValueError:
                                text = f"{series_label} as of {date_val}: {rate_val}%."
                            results.append({
                                "source": "FRED", "text": text,
                                "tier_weight": TIER_WEIGHTS["FRED"], "tier": 0,
                                "label": series_label,
                            })
            except Exception as e:
                logger.debug(f"FRED CSV ({series_id}) failed: {e}")

        if len(results) < 2:
            try:
                resp = requests.get(
                    "https://www.federalreserve.gov/feeds/press_all.xml",
                    headers=self.headers, timeout=6)
                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    for item in root.findall("./channel/item")[:2]:
                        title_elem = item.find("title")
                        desc_elem  = item.find("description")
                        if title_elem is not None and title_elem.text:
                            title    = self._clean_text(title_elem.text)
                            desc     = self._clean_text(desc_elem.text) if desc_elem is not None and desc_elem.text else ""
                            combined = f"{title}. {desc}"[:300]
                            if len(combined.strip()) > 10:
                                results.append({
                                    "source": "FRED", "text": combined,
                                    "tier_weight": TIER_WEIGHTS["FRED"], "tier": 0,
                                    "label": "Fed Press Release",
                                })
            except Exception as e:
                logger.debug(f"Fed Reserve RSS failed: {e}")
        return results[:SOURCE_MAX_ITEMS["FRED"]]

    # ==========================================================================
    # TIER 1: SEC EDGAR
    # ==========================================================================

    def fetch_sec_filings(self, ticker: str) -> List[Dict]:
        filings = []
        ticker_upper = ticker.upper()
        try:
            if ticker_upper in self.ticker_to_cik:
                cik = self.ticker_to_cik[ticker_upper]
                url = (f"https://www.sec.gov/cgi-bin/browse-edgar"
                       f"?action=getcompany&CIK={cik}&type=&dateb=&owner=exclude&count=5&output=atom")
            else:
                url = (f"https://www.sec.gov/cgi-bin/browse-edgar"
                       f"?company={ticker}&CIK=&action=getcompany&output=atom")
            resp = requests.get(url, headers=self.headers, timeout=5)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                ns   = {"atom": "http://www.w3.org/2005/Atom"}
                for entry in root.findall("atom:entry", ns)[:SOURCE_MAX_ITEMS["SEC EDGAR"]]:
                    title   = entry.find("atom:title", ns).text
                    summary = entry.find("atom:summary", ns)
                    summary = summary.text if summary is not None else ""
                    clean   = self._clean_text(f"{title} - {summary}")
                    filings.append({
                        "source": "SEC EDGAR", "text": clean,
                        "tier_weight": TIER_WEIGHTS["SEC EDGAR"], "tier": 1,
                    })
        except Exception as e:
            logger.debug(f"SEC EDGAR fetch failed: {e}")
        return filings

    # ==========================================================================
    # TIER 2: Yahoo Finance RSS
    # ==========================================================================

    def fetch_institutional_news(self, ticker: str) -> List[Dict]:
        news = []
        try:
            url  = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            resp = requests.get(url, headers=self.headers, timeout=5)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall("./channel/item")[:SOURCE_MAX_ITEMS["Yahoo Finance"]]:
                    title_elem = item.find("title")
                    if title_elem is not None and title_elem.text:
                        clean = self._clean_text(title_elem.text)
                        news.append({
                            "source": "Yahoo Finance", "text": clean,
                            "tier_weight": TIER_WEIGHTS["Yahoo Finance"], "tier": 2,
                        })
        except Exception as e:
            logger.debug(f"Yahoo Finance RSS failed: {e}")
        return news

    # ==========================================================================
    # TIER 3: GDELT
    # ==========================================================================

    def fetch_gdelt_geopolitical(self, ticker: str) -> List[Dict]:
        results = []
        topics  = self._gdelt_topics_for_ticker(ticker)
        for topic in topics[:2]:
            try:
                url = (
                    "https://api.gdeltproject.org/api/v2/doc/doc"
                    f"?query={requests.utils.quote(topic)}"
                    "&mode=artlist&maxrecords=5&format=json&timespan=1440&sort=DateDesc"
                )
                resp = requests.get(url, headers=self.headers, timeout=8)
                if resp.status_code == 200:
                    data = resp.json()
                    for article in data.get("articles", [])[:2]:
                        title = article.get("title", "")
                        if not title:
                            continue
                        clean_title = self._clean_text(title)
                        tone        = article.get("tone", None)
                        tone_text   = ""
                        if tone is not None:
                            try:
                                tv = float(tone)
                                if tv < -5:   tone_text = " Tone: highly negative geopolitical event."
                                elif tv < -2: tone_text = " Tone: negative geopolitical tension detected."
                                elif tv > 2:  tone_text = " Tone: positive diplomatic development."
                            except (ValueError, TypeError):
                                pass
                        results.append({
                            "source": "GDELT",
                            "text":   f"{clean_title}.{tone_text}"[:400],
                            "tier_weight": TIER_WEIGHTS["GDELT"], "tier": 3,
                            "topic": topic, "gdelt_tone": tone,
                        })
            except Exception as e:
                logger.debug(f"GDELT fetch failed for topic '{topic}': {e}")

        if len(results) < SOURCE_MAX_ITEMS["GDELT"]:
            conflict_topic = topics[0] + " conflict risk"
            try:
                conflict_url = (
                    "https://api.gdeltproject.org/api/v2/doc/doc"
                    f"?query={requests.utils.quote(conflict_topic)}"
                    "&mode=artlist&maxrecords=3&format=json&timespan=720&sort=DateDesc"
                )
                resp = requests.get(conflict_url, headers=self.headers, timeout=8)
                if resp.status_code == 200:
                    data = resp.json()
                    for article in data.get("articles", [])[:1]:
                        title = article.get("title", "")
                        if title:
                            results.append({
                                "source": "GDELT",
                                "text":   f"[CONFLICT ALERT] {self._clean_text(title)}",
                                "tier_weight": TIER_WEIGHTS["GDELT"] * 1.1, "tier": 3,
                                "topic": "conflict_monitor",
                                "gdelt_tone": article.get("tone"),
                            })
            except Exception as e:
                logger.debug(f"GDELT conflict monitor failed: {e}")
        return results[:SOURCE_MAX_ITEMS["GDELT"]]

    # ==========================================================================
    # TIER 4: Economic Calendar
    # ==========================================================================

    def fetch_economic_calendar(self, ticker: str) -> List[Dict]:
        results = []
        primary_urls = [
            "https://www.bls.gov/bls/news-release/rss.xml",
            "https://www.bea.gov/node/feed",
        ]
        for url in primary_urls:
            if results:
                break
            try:
                resp = requests.get(url, headers=self.headers, timeout=6)
                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    for item in root.findall("./channel/item")[:3]:
                        title_elem = item.find("title")
                        desc_elem  = item.find("description")
                        if title_elem is not None and title_elem.text:
                            title    = self._clean_text(title_elem.text)
                            desc     = self._clean_text(desc_elem.text) if desc_elem is not None and desc_elem.text else ""
                            combined = f"{title}. {desc}"[:300]
                            if len(combined.strip()) > 15:
                                results.append({
                                    "source": "EconCalendar", "text": combined,
                                    "tier_weight": TIER_WEIGHTS["EconCalendar"], "tier": 4,
                                })
            except Exception as e:
                logger.debug(f"EconCalendar primary RSS failed ({url}): {e}")

        if not results:
            try:
                resp = requests.get(
                    "https://tradingeconomics.com/rss/news.aspx",
                    headers=self.headers, timeout=6)
                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    for item in root.findall("./channel/item")[:3]:
                        title_elem = item.find("title")
                        if title_elem is not None and title_elem.text:
                            results.append({
                                "source": "EconCalendar",
                                "text":   self._clean_text(title_elem.text),
                                "tier_weight": TIER_WEIGHTS["EconCalendar"], "tier": 4,
                            })
            except Exception as e:
                logger.debug(f"EconCalendar tradingeconomics fallback failed: {e}")

        if not results:
            try:
                url  = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US"
                resp = requests.get(url, headers=self.headers, timeout=5)
                if resp.status_code == 200:
                    root   = ET.fromstring(resp.content)
                    macro_kw = ["CPI", "inflation", "GDP", "jobs", "Fed", "rate",
                                "PMI", "NFP", "payroll", "unemployment", "FOMC"]
                    for item in root.findall("./channel/item")[:8]:
                        title_elem = item.find("title")
                        if title_elem is not None and title_elem.text:
                            title = title_elem.text
                            if any(kw.lower() in title.lower() for kw in macro_kw):
                                results.append({
                                    "source": "EconCalendar",
                                    "text":   f"[MACRO] {self._clean_text(title)}",
                                    "tier_weight": TIER_WEIGHTS["EconCalendar"], "tier": 4,
                                })
            except Exception as e:
                logger.debug(f"EconCalendar Yahoo fallback failed: {e}")
        return results[:SOURCE_MAX_ITEMS["EconCalendar"]]

    # ==========================================================================
    # TIER 5: Macro FX & Commodity
    # ==========================================================================

    def fetch_macro_fx_commodity(self, ticker: str) -> List[Dict]:
        results = []
        numeric_symbols = {
            "^VIX":     ("VIX Volatility Index",   "fear index"),
            "^TNX":     ("US 10-Year Treasury Yield", "bond yield"),
            "DX-Y.NYB": ("US Dollar Index",         "DXY"),
        }
        for symbol, (label, short_name) in numeric_symbols.items():
            if len(results) >= SOURCE_MAX_ITEMS["MacroFX"]:
                break
            try:
                url  = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=2d"
                resp = requests.get(url, headers=self.headers, timeout=5)
                if resp.status_code == 200:
                    data   = resp.json()
                    closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
                    closes = [c for c in closes if c is not None]
                    if len(closes) >= 2:
                        current, prev  = closes[-1], closes[-2]
                        change_pct     = ((current - prev) / prev) * 100
                        if symbol == "^VIX":
                            if current > 25:
                                narrative = f"VIX fear index elevated at {current:.1f}, signaling high market anxiety."
                            elif current > 18:
                                narrative = f"VIX at {current:.1f}, moderate market concern. Change: {change_pct:+.1f}%."
                            else:
                                narrative = f"VIX low at {current:.1f}, markets calm and complacent."
                        else:
                            direction = "rising" if change_pct > 0 else "falling"
                            narrative = (f"{label} {direction} at {current:.2f} "
                                         f"({change_pct:+.2f}% vs prior session).")
                        results.append({
                            "source": "MacroFX", "text": narrative,
                            "tier_weight": TIER_WEIGHTS["MacroFX"], "tier": 5,
                            "symbol": symbol,
                        })
            except Exception as e:
                logger.debug(f"MacroFX numeric fetch failed for {symbol}: {e}")

        rss_symbols = {
            "GC=F":     "Gold Futures",
            "CL=F":     "WTI Crude Oil Futures",
            "EURUSD=X": "EUR/USD Exchange Rate",
        }
        for symbol, label in rss_symbols.items():
            if len(results) >= SOURCE_MAX_ITEMS["MacroFX"]:
                break
            try:
                url  = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
                resp = requests.get(url, headers=self.headers, timeout=5)
                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    for item in root.findall("./channel/item")[:1]:
                        title_elem = item.find("title")
                        if title_elem is not None and title_elem.text:
                            clean = self._clean_text(title_elem.text)
                            results.append({
                                "source": "MacroFX",
                                "text":   f"[{label}] {clean}",
                                "tier_weight": TIER_WEIGHTS["MacroFX"], "tier": 5,
                                "symbol": symbol,
                            })
            except Exception as e:
                logger.debug(f"MacroFX RSS failed for {symbol}: {e}")
        return results[:SOURCE_MAX_ITEMS["MacroFX"]]

    # ==========================================================================
    # TIER 6: Google Trends Proxy
    # ==========================================================================

    def fetch_google_trends_proxy(self, ticker: str) -> List[Dict]:
        results = []
        topics       = self._gdelt_topics_for_ticker(ticker)
        search_query = topics[0] if topics else ticker
        try:
            encoded_query = requests.utils.quote(search_query)
            url  = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
            resp = requests.get(url, headers=self.headers, timeout=6)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall("./channel/item")[:SOURCE_MAX_ITEMS["GoogleTrends"]]:
                    title_elem = item.find("title")
                    if title_elem is not None and title_elem.text:
                        clean = self._clean_text(title_elem.text)
                        clean = re.sub(r"\s+-\s+\S+$", "", clean)
                        if len(clean) > 10:
                            results.append({
                                "source": "GoogleTrends", "text": clean,
                                "tier_weight": TIER_WEIGHTS["GoogleTrends"], "tier": 6,
                            })
        except Exception as e:
            logger.debug(f"Google Trends proxy failed: {e}")
        return results

    # ==========================================================================
    # TIER 7: Reddit WallStreetBets
    # ==========================================================================

    def fetch_retail_momentum(self, ticker: str) -> List[Dict]:
        reddit_posts = []
        try:
            url  = (f"https://www.reddit.com/r/wallstreetbets/search.json"
                    f"?q={ticker}&restrict_sr=on&sort=new&limit=3")
            resp = requests.get(url, headers=self.reddit_headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                for child in data.get("data", {}).get("children", []):
                    post       = child["data"]
                    clean_text = self._clean_text(post.get("title", ""))
                    reddit_posts.append({
                        "source": "Reddit r/WSB", "text": clean_text,
                        "tier_weight": TIER_WEIGHTS["Reddit r/WSB"], "tier": 7,
                    })
        except Exception as e:
            logger.debug(f"Reddit fetch failed: {e}")
        return reddit_posts

    # ==========================================================================
    # TIER 8 (NEW v2.2): Future Event Scout
    # Returns upcoming events as structured text for LLM to reason about.
    # These are NOT scored by FinBERT HOLD they go directly to the LLM scorer
    # in sentiment_agent.py via the "future_events" key in the returned dict.
    # ==========================================================================

    def fetch_future_events(self, ticker: str) -> List[Dict]:
        """
        TIER 8 (weight 0.75): Upcoming event scanner.

        Sources:
          A. Next earnings date + expectations (Yahoo Finance calendar API)
          B. Next FOMC meeting (hardcoded 2026 schedule)
          C. Forward-looking news (Google News: "{ticker} upcoming next week")
          D. Reddit event mentions ("earnings play catalyst next week {ticker}")
          E. Economic release schedule (BLS advance release calendar)

        Items marked with "future_event": True so sentiment_agent.py can route
        them to the LLM scorer instead of FinBERT.
        """
        results = []

        # -- A: Earnings date -------------------------------------------------
        try:
            import yfinance as yf
            import io, sys, contextlib

            @contextlib.contextmanager
            def _s():
                old = sys.stdout, sys.stderr
                sys.stdout = sys.stderr = io.StringIO()
                try: yield
                finally: sys.stdout, sys.stderr = old

            with _s():
                ticker_obj = yf.Ticker(ticker)
                cal        = ticker_obj.calendar

            if cal is not None and not cal.empty:
                # calendar is a DataFrame with index = field names
                if "Earnings Date" in cal.index:
                    earnings_dates = cal.loc["Earnings Date"]
                    if hasattr(earnings_dates, '__iter__'):
                        dates_list = [d for d in earnings_dates if d is not None]
                    else:
                        dates_list = [earnings_dates]

                    if dates_list:
                        next_earnings = dates_list[0]
                        days_until    = (pd.Timestamp(next_earnings) - pd.Timestamp.now()).days \
                                        if hasattr(next_earnings, '__class__') else 0

                        eps_est = ""
                        if "EPS Estimate" in cal.index:
                            try:
                                eps_est = f" EPS estimate: ${float(cal.loc['EPS Estimate'].iloc[0]):.2f}."
                            except Exception:
                                pass

                        rev_est = ""
                        if "Revenue Estimate" in cal.index:
                            try:
                                rev_val = float(cal.loc["Revenue Estimate"].iloc[0])
                                rev_est = f" Revenue estimate: ${rev_val/1e9:.1f}B."
                            except Exception:
                                pass

                        if days_until >= 0:
                            text = (f"[UPCOMING EARNINGS] {ticker} reports earnings on "
                                    f"{str(next_earnings)[:10]} ({days_until} days away)."
                                    f"{eps_est}{rev_est} Markets will react to beat/miss.")
                        else:
                            text = (f"[RECENT EARNINGS] {ticker} reported earnings on "
                                    f"{str(next_earnings)[:10]}. Post-earnings reaction ongoing.")

                        results.append({
                            "source":       "FutureEvents",
                            "text":         text,
                            "tier_weight":  TIER_WEIGHTS["FutureEvents"],
                            "tier":         8,
                            "future_event": True,
                            "event_type":   "earnings",
                            "days_until":   int(days_until) if isinstance(days_until, (int, float)) else 0,
                        })
        except Exception as e:
            logger.debug(f"Future: Earnings calendar fetch failed for {ticker}: {e}")

        # -- B: Next FOMC meeting ----------------------------------------------
        try:
            today = datetime.today().date()
            next_fomc = None
            for start_str, end_str in FOMC_2026_DATES:
                meeting_start = datetime.strptime(start_str, "%Y-%m-%d").date()
                meeting_end   = datetime.strptime(end_str,   "%Y-%m-%d").date()
                if meeting_start >= today:
                    next_fomc = (meeting_start, meeting_end)
                    break

            if next_fomc:
                days_until = (next_fomc[0] - today).days
                if days_until <= 30:   # only if within next 30 days
                    text = (f"[UPCOMING FOMC] Federal Reserve FOMC meeting scheduled "
                            f"{next_fomc[0]} to {next_fomc[1]} ({days_until} days away). "
                            f"Rate decision will impact growth stocks, bonds, and sector rotations. "
                            f"Market positioning before and after meeting can cause volatility.")
                    results.append({
                        "source":       "FutureEvents",
                        "text":         text,
                        "tier_weight":  TIER_WEIGHTS["FutureEvents"],
                        "tier":         8,
                        "future_event": True,
                        "event_type":   "fomc",
                        "days_until":   days_until,
                    })
        except Exception as e:
            logger.debug(f"Future: FOMC calendar failed: {e}")

        # -- C: Forward-looking news (Google News) -----------------------------
        forward_queries = [
            f"{ticker} upcoming announcement next week",
            f"{ticker} product launch event 2026",
            f"{ticker} earnings catalyst upcoming",
        ]
        for query in forward_queries[:2]:
            if len(results) >= SOURCE_MAX_ITEMS["FutureEvents"] - 1:
                break
            try:
                encoded = requests.utils.quote(query)
                url     = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
                resp    = requests.get(url, headers=self.headers, timeout=6)
                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    for item in root.findall("./channel/item")[:2]:
                        title_elem = item.find("title")
                        pub_date   = item.find("pubDate")
                        if title_elem is not None and title_elem.text:
                            clean = self._clean_text(title_elem.text)
                            clean = re.sub(r"\s+-\s+\S+$", "", clean)
                            # FIX-3: Two-gate filter:
                            # Gate 1 HOLD must contain forward-looking intent keyword
                            # Gate 2 HOLD must pass finance relevance (no Stranger Things etc.)
                            fwd_kw = ["upcoming", "next", "plan", "launch", "announce",
                                      "schedule", "expect", "forecast", "will", "event",
                                      "preview", "ahead", "before", "catalyst", "earnings"]
                            has_fwd      = any(kw in clean.lower() for kw in fwd_kw)
                            is_finance   = _is_finance_relevant(clean)
                            if has_fwd and is_finance and len(clean) > 15:
                                results.append({
                                    "source":       "FutureEvents",
                                    "text":         f"[UPCOMING NEWS] {clean}",
                                    "tier_weight":  TIER_WEIGHTS["FutureEvents"],
                                    "tier":         8,
                                    "future_event": True,
                                    "event_type":   "news_forward",
                                })
            except Exception as e:
                logger.debug(f"Future: Google forward news failed for '{query}': {e}")

        # -- D: Reddit upcoming event mentions ---------------------------------
        try:
            url  = (f"https://www.reddit.com/r/wallstreetbets/search.json"
                    f"?q={ticker}+earnings+play+catalyst&restrict_sr=on&sort=new&limit=3")
            resp = requests.get(url, headers=self.reddit_headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                for child in data.get("data", {}).get("children", [])[:2]:
                    post  = child["data"]
                    title = self._clean_text(post.get("title", ""))
                    fwd_kw = ["earnings", "catalyst", "play", "next week", "upcoming",
                               "before earnings", "yolo", "calls", "puts", "event"]
                    # FIX-3: Reddit posts must also pass finance relevance gate
                    has_fwd    = any(kw in title.lower() for kw in fwd_kw)
                    is_finance = _is_finance_relevant(title)
                    if has_fwd and is_finance and len(title) > 10:
                        score = post.get("score", 0)
                        text  = f"[RETAIL CATALYST] {title} (upvotes: {score})"
                        results.append({
                            "source":       "FutureEvents",
                            "text":         text,
                            "tier_weight":  TIER_WEIGHTS["FutureEvents"] * 0.6,
                            "tier":         8,
                            "future_event": True,
                            "event_type":   "reddit_catalyst",
                        })
        except Exception as e:
            logger.debug(f"Future: Reddit catalyst scan failed: {e}")

        # -- E: BLS advance release schedule -----------------------------------
        try:
            url  = "https://www.bls.gov/bls/news-release/rss.xml"
            resp = requests.get(url, headers=self.headers, timeout=6)
            if resp.status_code == 200:
                root   = ET.fromstring(resp.content)
                fwd_kw = ["advance", "preliminary", "scheduled", "upcoming",
                          "next release", "will release", "estimate"]
                for item in root.findall("./channel/item")[:5]:
                    title_elem = item.find("title")
                    if title_elem is not None and title_elem.text:
                        title = title_elem.text
                        if any(kw in title.lower() for kw in fwd_kw):
                            clean = self._clean_text(title)
                            results.append({
                                "source":       "FutureEvents",
                                "text":         f"[BLS UPCOMING] {clean}",
                                "tier_weight":  TIER_WEIGHTS["FutureEvents"] * 0.8,
                                "tier":         8,
                                "future_event": True,
                                "event_type":   "economic_release",
                            })
        except Exception as e:
            logger.debug(f"Future: BLS advance schedule failed: {e}")

        return results[:SOURCE_MAX_ITEMS["FutureEvents"]]

    # ==========================================================================
    # MASTER ASSEMBLER
    # ==========================================================================

    def get_global_context_payload(self, ticker: str) -> List[Dict]:
        """
        Assembles the full 9-tier intelligence payload.

        Tiers 0-7: Present-day sentiment (FinBERT scored in sentiment_agent.py)
        Tier 8:    Future events (LLM scored in sentiment_agent.py)

        Items with "future_event": True are routed to the LLM scorer.
        Items without it go to FinBERT as before.
        """
        print(f"      📡 [MCP v2.2] Broadcasting across 9 tiers for {ticker}...")

        payload: List[Dict] = []

        payload.extend(self.fetch_fed_macro(ticker))            # Tier 0: FRED
        payload.extend(self.fetch_sec_filings(ticker))          # Tier 1: SEC
        payload.extend(self.fetch_institutional_news(ticker))   # Tier 2: Yahoo
        payload.extend(self.fetch_gdelt_geopolitical(ticker))   # Tier 3: GDELT
        payload.extend(self.fetch_economic_calendar(ticker))    # Tier 4: EconCal
        payload.extend(self.fetch_macro_fx_commodity(ticker))   # Tier 5: MacroFX
        payload.extend(self.fetch_google_trends_proxy(ticker))  # Tier 6: GTrends
        payload.extend(self.fetch_retail_momentum(ticker))      # Tier 7: Reddit
        payload.extend(self.fetch_future_events(ticker))        # Tier 8: Future ← NEW

        # Fuzzy deduplication (first 60 chars)
        seen_fingerprints: set = set()
        clean_payload: List[Dict] = []
        for item in payload:
            text = item.get("text", "")
            if len(text.strip()) <= 5:
                continue
            fingerprint = text.strip().lower()[:60]
            if fingerprint not in seen_fingerprints:
                clean_payload.append(item)
                seen_fingerprints.add(fingerprint)

        # Sort by tier_weight descending
        clean_payload.sort(key=lambda x: x.get("tier_weight", 0), reverse=True)

        # Global cap at 30 items (raised from 25 to accommodate Tier 8)
        clean_payload = clean_payload[:30]

        if not clean_payload:
            clean_payload.append({
                "source":      "System Fallback",
                "text":        f"{ticker} trading in standard market conditions with low news velocity.",
                "tier_weight": 0.50,
                "tier":        99,
            })

        tier_counts = {}
        for item in clean_payload:
            src = item.get("source", "Unknown")
            tier_counts[src] = tier_counts.get(src, 0) + 1

        future_count = sum(1 for i in clean_payload if i.get("future_event"))
        print(f"      [OK] [MCP v2.2] Assembly complete. "
              f"{len(clean_payload)} signals ({future_count} future events).")
        print(f"         Sources: {tier_counts}")

        return clean_payload


# -- import needed by fetch_future_events -------------------------------------
try:
    import pandas as pd
except ImportError:
    pass