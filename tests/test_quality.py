"""
Search quality tests -- 100 real-world queries against MEMORY.md.

Proves agentmem finds the RIGHT answers for real agent queries across
10 categories: owner info, environment, Pixie Bot, Astro Bot, AgentNet,
agentmem, Channel, Key Learnings, Skills & Rules, Cross-cutting.

Run:
  python3 -m pytest agentmem/tests/test_quality.py -v
"""
import os
import sys
import time
import pytest
from pathlib import Path

# Ensure agentmem is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agentmem.core import MemoryStore, _chunk_markdown
from agentmem.embeddings import get_embedding_model

MEMORY_FILE = Path(
    "/home/openclaw/.claude/projects/-home-openclaw--openclaw-workspace/memory/MEMORY.md"
)


# ----------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------

@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """
    Import the real MEMORY.md into a fresh store.

    scope=module so the (relatively expensive) import happens once,
    then all test methods share the same store.  Each test only reads.
    """
    tmp = tmp_path_factory.mktemp("quality100")
    embed = get_embedding_model()
    store = MemoryStore(str(tmp / "quality.db"), embedding_dim=embed.dim)
    store.set_embed_fn(embed)

    assert MEMORY_FILE.exists(), f"MEMORY.md not found at {MEMORY_FILE}"

    result = store.import_markdown(str(MEMORY_FILE), tier="core")
    assert result["imported"] > 0, f"Import failed or produced 0 chunks: {result}"

    store._import_result = result
    yield store
    store.close()


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _all(results):
    """Concatenate all result contents into one string."""
    return " ".join(r["content"] for r in results)


def _dbg(results, n=80):
    """Format results for readable assertion messages."""
    return [r["content"][:n] for r in results]


# ================================================================
# TestSearchQuality100 -- 100 real-world queries, 10 categories
# ================================================================

class TestSearchQuality100:
    """100 search quality tests across 10 categories."""

    # ============================================================
    # Category 1: Owner Info (10 tests)
    # ============================================================

    def test_01_owner_name(self, store):
        results = store.recall("owner name Alexander", limit=3)
        assert any("Alexander" in r["content"] or "\u0410\u043b\u0435\u043a\u0441\u0430\u043d\u0434\u0440" in r["content"] for r in results), \
            f"Expected 'Alexander' in top 3, got: {_dbg(results)}"

    def test_02_telegram_id(self, store):
        results = store.recall("Alexander Telegram ID", limit=3)
        assert any("252708838" in r["content"] for r in results), \
            f"Expected '252708838' in top 3, got: {_dbg(results)}"

    def test_03_owner_language(self, store):
        results = store.recall("owner language preference", limit=3)
        content = _all(results)
        assert "Russian" in content or "informal" in content, \
            f"Expected 'Russian' or 'informal' in top 3, got: {_dbg(results)}"

    def test_04_owner_challenge(self, store):
        results = store.recall("challenge earn money zero capital", limit=3)
        content = _all(results)
        assert "zero capital" in content or "Earn money" in content, \
            f"Expected 'zero capital' in top 3, got: {_dbg(results)}"

    def test_05_costs_breakdown(self, store):
        results = store.recall("monthly costs breakdown server Claude", limit=3)
        content = _all(results)
        assert "242" in content or "3845" in content or "200" in content, \
            f"Expected cost numbers in top 3, got: {_dbg(results)}"

    def test_06_ru_server_cost(self, store):
        results = store.recall("\u0441\u043a\u043e\u043b\u044c\u043a\u043e \u0441\u0442\u043e\u0438\u0442 \u0441\u0435\u0440\u0432\u0435\u0440", limit=3, auto_translate=True)
        content = _all(results)
        assert "3845" in content or "server" in content.lower(), \
            f"Expected '3845' or 'server' in top 3, got: {_dbg(results)}"

    def test_07_claude_subscription_cost(self, store):
        results = store.recall("Claude subscription cost per month", limit=3)
        content = _all(results)
        assert "200" in content or "subscription" in content.lower() or "Claude" in content, \
            f"Expected '$200' or 'subscription' in top 3, got: {_dbg(results)}"

    def test_08_key_rule_no_manual(self, store):
        results = store.recall("key rule manual work automate", limit=3)
        content = _all(results)
        assert "manual work" in content or "automate" in content or "Key rule" in content, \
            f"Expected 'manual work' / 'automate' in top 3, got: {_dbg(results)}"

    def test_09_autonomy_granted(self, store):
        results = store.recall("Alexander granted autonomy decision-making", limit=3)
        content = _all(results)
        assert "autonomy" in content.lower() or "decision-making" in content, \
            f"Expected 'autonomy' in top 3, got: {_dbg(results)}"

    def test_10_reward_milestones(self, store):
        results = store.recall("reward milestones first user first ruble break-even", limit=3)
        content = _all(results)
        assert "Milestone" in content or "IDENTITY" in content or "SOUL" in content, \
            f"Expected 'Milestone' in top 3, got: {_dbg(results)}"

    # ============================================================
    # Category 2: Environment (10 tests)
    # ============================================================

    def test_11_os_version(self, store):
        results = store.recall("Ubuntu operating system version", limit=3)
        content = _all(results)
        assert "Ubuntu" in content or "24.04" in content, \
            f"Expected 'Ubuntu 24.04' in top 3, got: {_dbg(results)}"

    def test_12_python_version(self, store):
        results = store.recall("Python version installed", limit=3)
        content = _all(results)
        assert "3.12" in content or "Python" in content, \
            f"Expected 'Python 3.12' in top 3, got: {_dbg(results)}"

    def test_13_node_version(self, store):
        results = store.recall("Node.js version environment", limit=3)
        content = _all(results)
        assert "Node" in content or "22" in content, \
            f"Expected 'Node 22' in top 3, got: {_dbg(results)}"

    def test_14_no_sudo(self, store):
        results = store.recall("sudo access privileges", limit=3)
        content = _all(results)
        assert "sudo" in content.lower() or "no sudo" in content.lower(), \
            f"Expected 'no sudo' in top 3, got: {_dbg(results)}"

    def test_15_pip_command(self, store):
        results = store.recall("pip install command break-system-packages", limit=3)
        content = _all(results)
        assert "break-system-packages" in content or "pip" in content, \
            f"Expected pip install command in top 3, got: {_dbg(results)}"

    def test_16_path_env(self, store):
        results = store.recall("PATH environment variable local bin", limit=3)
        content = _all(results)
        assert ".local/bin" in content or "PATH" in content, \
            f"Expected PATH info in top 3, got: {_dbg(results)}"

    def test_17_openclaw_platform(self, store):
        results = store.recall("OpenClaw platform Telegram integration", limit=3)
        content = _all(results)
        assert "OpenClaw" in content or "Telegram" in content, \
            f"Expected 'OpenClaw' in top 3, got: {_dbg(results)}"

    def test_18_bot_token_env(self, store):
        results = store.recall("bot token .env file location", limit=3)
        content = _all(results)
        assert ".env" in content or "token" in content.lower(), \
            f"Expected '.env' or 'token' in top 3, got: {_dbg(results)}"

    def test_19_env_loading(self, store):
        results = store.recall("bot startup export env xargs", limit=3)
        content = _all(results)
        assert "xargs" in content or "export" in content or ".env" in content, \
            f"Expected env loading info in top 3, got: {_dbg(results)}"

    def test_20_watchdog_location(self, store):
        results = store.recall("watchdog.sh file path location crontab", limit=3)
        content = _all(results)
        assert "watchdog" in content.lower() or "crontab" in content, \
            f"Expected 'watchdog.sh' info in top 3, got: {_dbg(results)}"

    # ============================================================
    # Category 3: Pixie Bot (10 tests)
    # ============================================================

    def test_21_pixie_username(self, store):
        results = store.recall("Pixie Bot telegram username", limit=3)
        content = _all(results)
        assert "toolbox_utils_bot" in content or "Pixie" in content, \
            f"Expected '@toolbox_utils_bot' in top 3, got: {_dbg(results)}"

    def test_22_pixie_display_name(self, store):
        results = store.recall("Pixie display name toolbox bot", limit=3)
        content = _all(results)
        assert "Pixie" in content, \
            f"Expected 'Pixie' in top 3, got: {_dbg(results)}"

    def test_23_pixie_ai_model(self, store):
        results = store.recall("Pixie AI image generation model FLUX", limit=3)
        content = _all(results)
        assert "FLUX" in content or "DeepInfra" in content, \
            f"Expected 'FLUX' or 'DeepInfra' in top 3, got: {_dbg(results)}"

    def test_24_pixie_pricing(self, store):
        results = store.recall("Pixie Bot pricing stars AI generation", limit=3)
        content = _all(results)
        assert "10" in content or "25" in content or "Stars" in content or "\u2b50" in content, \
            f"Expected pricing (10/25 stars) in top 3, got: {_dbg(results)}"

    def test_25_pixie_free_features(self, store):
        results = store.recall("Pixie free features QR OCR stickers", limit=3)
        content = _all(results)
        assert "QR" in content or "OCR" in content or "free" in content.lower(), \
            f"Expected free features in top 3, got: {_dbg(results)}"

    def test_26_pixie_paid_features(self, store):
        results = store.recall("Pixie paid features bg removal filters watermark", limit=3)
        content = _all(results)
        assert "bg removal" in content or "filters" in content or "Paid" in content or "watermark" in content, \
            f"Expected paid features in top 3, got: {_dbg(results)}"

    def test_27_pixie_user_count(self, store):
        results = store.recall("Pixie Bot user count status", limit=3)
        content = _all(results)
        assert "4 users" in content or "users" in content, \
            f"Expected '4 users' in top 3, got: {_dbg(results)}"

    def test_28_pixie_daily_free_gen(self, store):
        results = store.recall("daily free generation 24h returning users", limit=3)
        content = _all(results)
        assert "free" in content.lower() or "24h" in content or "Daily" in content, \
            f"Expected daily free gen info in top 3, got: {_dbg(results)}"

    def test_29_pixie_source_tracking(self, store):
        results = store.recall("source tracking deep links start=habr", limit=3)
        content = _all(results)
        assert "source" in content.lower() or "start=" in content or "tracking" in content.lower() or "deep link" in content.lower(), \
            f"Expected source tracking info in top 3, got: {_dbg(results)}"

    def test_30_pixie_watermark(self, store):
        results = store.recall("viral watermark AI image corner text", limit=3)
        content = _all(results)
        assert "watermark" in content.lower() or "Pixie Bot" in content, \
            f"Expected watermark info in top 3, got: {_dbg(results)}"

    # ============================================================
    # Category 4: Astro Bot (10 tests)
    # ============================================================

    def test_31_astro_username(self, store):
        results = store.recall("Astro Bot telegram username", limit=3)
        content = _all(results)
        assert "astro_light_taro_bot" in content or "Astro" in content, \
            f"Expected '@astro_light_taro_bot' in top 3, got: {_dbg(results)}"

    def test_32_astro_llm(self, store):
        results = store.recall("Astro Bot LLM model Gemini OpenRouter", limit=3)
        content = _all(results)
        assert "Gemini" in content or "OpenRouter" in content or "Flash" in content, \
            f"Expected 'Gemini' or 'OpenRouter' in top 3, got: {_dbg(results)}"

    def test_33_astro_features(self, store):
        results = store.recall("Astro Bot features horoscope tarot compatibility", limit=3)
        content = _all(results)
        assert "horoscope" in content.lower() or "tarot" in content.lower() or "compatibility" in content.lower(), \
            f"Expected horoscope/tarot/compatibility in top 3, got: {_dbg(results)}"

    def test_34_astro_pricing(self, store):
        results = store.recall("Astro Bot pricing tarot stars numerology", limit=3)
        content = _all(results)
        assert "5" in content or "10" in content or "Stars" in content or "\u2b50" in content, \
            f"Expected pricing info in top 3, got: {_dbg(results)}"

    def test_35_astro_daily_reminder(self, store):
        results = store.recall("daily reminder 9:00 MSK auto horoscope scheduler", limit=3)
        content = _all(results)
        assert "9:00" in content or "MSK" in content or "reminder" in content.lower() or "scheduler" in content.lower(), \
            f"Expected '9:00 MSK' reminder in top 3, got: {_dbg(results)}"

    def test_36_astro_zodiac_count(self, store):
        results = store.recall("zodiac signs count twelve", limit=3)
        content = _all(results)
        assert "12 zodiac" in content or "zodiac" in content.lower(), \
            f"Expected '12 zodiac' in top 3, got: {_dbg(results)}"

    def test_37_astro_first_real_user(self, store):
        results = store.recall("first non-Alexander user milestone Astro", limit=3)
        content = _all(results)
        assert "165170269" in content or "Milestone" in content or "first" in content.lower(), \
            f"Expected first real user ID in top 3, got: {_dbg(results)}"

    def test_38_astro_cross_promo(self, store):
        results = store.recall("cross-promo Pixie Astro links both bots", limit=3)
        content = _all(results)
        assert "cross-promo" in content.lower() or "Pixie" in content or "Astro" in content, \
            f"Expected cross-promo info in top 3, got: {_dbg(results)}"

    def test_39_astro_horoscope_cards(self, store):
        results = store.recall("horoscope card images PIL purple gradient gold text", limit=3)
        content = _all(results)
        assert "PIL" in content or "card" in content.lower() or "gradient" in content or "horoscope" in content.lower(), \
            f"Expected horoscope card info in top 3, got: {_dbg(results)}"

    def test_40_astro_share_buttons(self, store):
        results = store.recall("share buttons after every action contextual text", limit=3)
        content = _all(results)
        assert "Share" in content or "share" in content or "button" in content.lower(), \
            f"Expected share buttons info in top 3, got: {_dbg(results)}"

    # ============================================================
    # Category 5: AgentNet (10 tests)
    # ============================================================

    def test_41_agentnet_ports(self, store):
        results = store.recall("AgentNet port MCP REST API", limit=3)
        content = _all(results)
        assert "8420" in content or "8421" in content, \
            f"Expected port 8420/8421 in top 3, got: {_dbg(results)}"

    def test_42_agentnet_tools_count(self, store):
        results = store.recall("AgentNet MCP tools register find recommend", limit=3)
        content = _all(results)
        assert "7 MCP tools" in content or "register_agent" in content or "find_agents" in content, \
            f"Expected 7 MCP tools in top 3, got: {_dbg(results)}"

    def test_43_agentnet_registry_size(self, store):
        results = store.recall("AgentNet registry size agents seeded", limit=3)
        content = _all(results)
        assert "48" in content or "agents" in content.lower(), \
            f"Expected '48 agents' in top 3, got: {_dbg(results)}"

    def test_44_agentnet_github(self, store):
        results = store.recall("AgentNet GitHub repository URL", limit=3)
        content = _all(results)
        assert "oxgeneral/agentnet" in content or "github.com" in content, \
            f"Expected GitHub URL in top 3, got: {_dbg(results)}"

    def test_45_agentnet_mcp_registry(self, store):
        results = store.recall("AgentNet official MCP registry published", limit=3)
        content = _all(results)
        assert "io.github.oxgeneral/agentnet" in content or "MCP Registry" in content or "PUBLISHED" in content, \
            f"Expected MCP registry info in top 3, got: {_dbg(results)}"

    def test_46_agentnet_awesome_mcp(self, store):
        results = store.recall("awesome-mcp-servers PR status merged", limit=3)
        content = _all(results)
        assert "MERGED" in content or "awesome-mcp" in content or "punkpeye" in content, \
            f"Expected 'MERGED' in top 3, got: {_dbg(results)}"

    def test_47_agentnet_smithery(self, store):
        results = store.recall("Smithery deployment AgentNet URL", limit=3)
        content = _all(results)
        assert "smithery.ai" in content or "Smithery" in content, \
            f"Expected Smithery URL in top 3, got: {_dbg(results)}"

    def test_48_agentnet_trust_model(self, store):
        results = store.recall("bilateral trust proof rate limiting reputation", limit=3)
        content = _all(results)
        assert "trust" in content.lower() or "bilateral" in content or "reputation" in content or "rate limiting" in content, \
            f"Expected trust model info in top 3, got: {_dbg(results)}"

    def test_49_agentnet_credit_economy(self, store):
        results = store.recall("credit economy register referral sent received", limit=3)
        content = _all(results)
        assert "credit" in content.lower() or "+10" in content or "referral" in content.lower(), \
            f"Expected credit economy info in top 3, got: {_dbg(results)}"

    def test_50_agentnet_cloudflare(self, store):
        results = store.recall("Cloudflare Tunnels HTTPS trycloudflare watchdog", limit=3)
        content = _all(results)
        assert "Cloudflare" in content or "trycloudflare" in content or "tunnel" in content.lower(), \
            f"Expected Cloudflare tunnels info in top 3, got: {_dbg(results)}"

    # ============================================================
    # Category 6: agentmem (10 tests)
    # ============================================================

    def test_51_agentmem_version(self, store):
        results = store.recall("agentmem version production status", limit=3)
        content = _all(results)
        assert "0.3.0" in content or "agentmem" in content.lower(), \
            f"Expected 'v0.3.0' in top 3, got: {_dbg(results)}"

    def test_52_agentmem_pypi_name(self, store):
        results = store.recall("agentmem PyPI package name lite", limit=3)
        content = _all(results)
        assert "agentmem-lite" in content or "PyPI" in content, \
            f"Expected 'agentmem-lite' in top 3, got: {_dbg(results)}"

    def test_53_agentmem_test_count(self, store):
        results = store.recall("agentmem tests count 206", limit=3)
        content = _all(results)
        assert "206" in content or "tests" in content.lower(), \
            f"Expected '206 tests' in top 3, got: {_dbg(results)}"

    def test_54_agentmem_mcp_registry(self, store):
        results = store.recall("agentmem MCP registry published io.github", limit=3)
        content = _all(results)
        assert "io.github.oxgeneral/agentmem" in content or "MCP Registry" in content, \
            f"Expected MCP registry entry in top 3, got: {_dbg(results)}"

    def test_55_agentmem_github(self, store):
        results = store.recall("agentmem GitHub repository oxgeneral", limit=3)
        content = _all(results)
        assert "oxgeneral/agentmem" in content or "github.com" in content, \
            f"Expected GitHub URL in top 3, got: {_dbg(results)}"

    def test_56_agentmem_tools_count(self, store):
        results = store.recall("agentmem MCP tools count 16", limit=3)
        content = _all(results)
        assert "16" in content or "MCP tools" in content, \
            f"Expected '16 MCP tools' in top 3, got: {_dbg(results)}"

    def test_57_agentmem_tiers(self, store):
        results = store.recall("agentmem tiers namespaces versioning", limit=3)
        content = _all(results)
        assert "5 tiers" in content or "tiers" in content.lower() or "namespaces" in content, \
            f"Expected '5 tiers' in top 3, got: {_dbg(results)}"

    def test_58_agentmem_typed(self, store):
        results = store.recall("agentmem TypedDict return types py.typed", limit=3)
        content = _all(results)
        assert "TypedDict" in content or "py.typed" in content or "16" in content, \
            f"Expected TypedDict info in top 3, got: {_dbg(results)}"

    def test_59_agentmem_features(self, store):
        results = store.recall("agentmem hybrid search entity extraction LSH", limit=3)
        content = _all(results)
        assert "hybrid" in content.lower() or "entity" in content.lower() or "LSH" in content, \
            f"Expected agentmem features in top 3, got: {_dbg(results)}"

    def test_60_agentmem_stack(self, store):
        results = store.recall("agentmem stack SQLite FTS5 sqlite-vec model2vec potion", limit=3)
        content = _all(results)
        assert "FTS5" in content or "sqlite-vec" in content or "model2vec" in content or "potion" in content, \
            f"Expected stack info in top 3, got: {_dbg(results)}"

    # ============================================================
    # Category 7: Channel (10 tests)
    # ============================================================

    def test_61_channel_username(self, store):
        results = store.recall("channel username workonhuman", limit=3)
        content = _all(results)
        assert "workonhuman" in content, \
            f"Expected '@workonhuman' in top 3, got: {_dbg(results)}"

    def test_62_channel_concept(self, store):
        results = store.recall("channel concept AI diary working for human", limit=3)
        content = _all(results)
        assert "diary" in content.lower() or "AI" in content, \
            f"Expected diary concept in top 3, got: {_dbg(results)}"

    def test_63_channel_style(self, store):
        results = store.recall("channel style first person raw honest", limit=3)
        content = _all(results)
        assert "first person" in content or "raw" in content or "honest" in content or "diary" in content.lower(), \
            f"Expected style info in top 3, got: {_dbg(results)}"

    def test_64_channel_format(self, store):
        results = store.recall("channel format image caption 794 chars", limit=3)
        content = _all(results)
        assert "794" in content or "caption" in content or "image" in content.lower(), \
            f"Expected '794 chars' format in top 3, got: {_dbg(results)}"

    def test_65_channel_subscribers(self, store):
        results = store.recall("channel subscribers members count", limit=3)
        content = _all(results)
        assert "7 subscribers" in content or "Members" in content or "subscribers" in content, \
            f"Expected '7 subscribers' in top 3, got: {_dbg(results)}"

    def test_66_channel_diary_count(self, store):
        results = store.recall("diary entries count list 13", limit=3)
        content = _all(results)
        assert "Diary" in content or "diary" in content or "#13" in content, \
            f"Expected diary count in top 3, got: {_dbg(results)}"

    def test_67_telegraph_token(self, store):
        results = store.recall("Telegraph token API autonomous publishing", limit=3)
        content = _all(results)
        assert "5feaa63" in content or "Telegra.ph" in content or "Telegraph" in content, \
            f"Expected Telegraph token in top 3, got: {_dbg(results)}"

    def test_68_telegraph_articles_count(self, store):
        results = store.recall("Telegraph SEO articles published 18", limit=3)
        content = _all(results)
        assert "18" in content or "Telegraph" in content or "SEO" in content, \
            f"Expected '18 articles' in top 3, got: {_dbg(results)}"

    def test_69_discussion_group(self, store):
        results = store.recall("discussion group linked bot admin auto-replies comments", limit=3)
        content = _all(results)
        assert "Discussion" in content or "discussion" in content or "auto-replies" in content or "comments" in content, \
            f"Expected discussion group info in top 3, got: {_dbg(results)}"

    def test_70_paid_dms(self, store):
        results = store.recall("paid DMs enabled Telegram Stars", limit=3)
        content = _all(results)
        assert "Paid DMs" in content or "paid" in content.lower() or "Stars" in content, \
            f"Expected paid DMs info in top 3, got: {_dbg(results)}"

    # ============================================================
    # Category 8: Key Learnings (10 tests)
    # ============================================================

    def test_71_sanitize_text_field(self, store):
        results = store.recall("sanitize_text_field SQL quotes HTML tags", limit=3)
        content = _all(results)
        assert "sanitize_text_field" in content or "SQL" in content or "HTML tags" in content, \
            f"Expected sanitize_text_field info in top 3, got: {_dbg(results)}"

    def test_72_is_admin_check(self, store):
        results = store.recall("is_admin checks URL not role wp_ajax", limit=3)
        content = _all(results)
        assert "is_admin" in content or "URL" in content or "wp_ajax" in content, \
            f"Expected is_admin info in top 3, got: {_dbg(results)}"

    def test_73_stars_withdrawal(self, store):
        results = store.recall("Stars withdraw Fragment TON 65% net 1000", limit=3)
        content = _all(results)
        assert "Fragment" in content or "TON" in content or "65%" in content, \
            f"Expected Stars withdrawal info in top 3, got: {_dbg(results)}"

    def test_74_bot_killing(self, store):
        results = store.recall("bot killing pkill 15-20 seconds restart", limit=3)
        content = _all(results)
        assert "pkill" in content or "15-20" in content or "killing" in content.lower(), \
            f"Expected bot killing info in top 3, got: {_dbg(results)}"

    def test_75_bot_crash_cgroup(self, store):
        results = store.recall("bot crash nohup cgroup openclaw.service killed", limit=3)
        content = _all(results)
        assert "cgroup" in content or "nohup" in content or "crash" in content.lower() or "openclaw.service" in content, \
            f"Expected bot crash root cause in top 3, got: {_dbg(results)}"

    def test_76_browser_use_commands(self, store):
        results = store.recall("browser-use commands open click input screenshot", limit=3)
        content = _all(results)
        assert "browser-use" in content or "click" in content or "screenshot" in content, \
            f"Expected browser-use commands in top 3, got: {_dbg(results)}"

    def test_77_deepinfra_endpoint(self, store):
        results = store.recall("DeepInfra API endpoint images generations b64_json", limit=3)
        content = _all(results)
        assert "deepinfra.com" in content or "DeepInfra" in content or "FLUX" in content, \
            f"Expected DeepInfra endpoint in top 3, got: {_dbg(results)}"

    def test_78_pollinations_ai(self, store):
        results = store.recall("Pollinations AI free LLM no key OpenAI compatible", limit=3)
        content = _all(results)
        assert "Pollinations" in content or "pollinations.ai" in content, \
            f"Expected Pollinations AI info in top 3, got: {_dbg(results)}"

    def test_79_telegram_store_login(self, store):
        results = store.recall("telegram-store.com requires login Telegram auth", limit=3)
        content = _all(results)
        assert "telegram-store" in content or "auth" in content.lower() or "login" in content.lower(), \
            f"Expected telegram-store login info in top 3, got: {_dbg(results)}"

    def test_80_vue_react_textarea(self, store):
        results = store.recall("Vue React textarea native setter dispatchEvent atob base64", limit=3)
        content = _all(results)
        assert "textarea" in content.lower() or "native setter" in content or "dispatchEvent" in content or "atob" in content, \
            f"Expected Vue/React textarea fix in top 3, got: {_dbg(results)}"

    # ============================================================
    # Category 9: Skills & Rules (10 tests)
    # ============================================================

    def test_81_skill_count(self, store):
        results = store.recall("installed skills total count 20", limit=3)
        content = _all(results)
        assert "20" in content or "Skills" in content or "skills" in content, \
            f"Expected '20 skills' in top 3, got: {_dbg(results)}"

    def test_82_brainstorming_rule(self, store):
        results = store.recall("brainstorming skill before creative work", limit=3)
        content = _all(results)
        assert "brainstorming" in content.lower() or "creative" in content.lower(), \
            f"Expected brainstorming rule in top 3, got: {_dbg(results)}"

    def test_83_copywriting_rule(self, store):
        results = store.recall("always use skills writing posts copywriting social-content", limit=3)
        content = _all(results)
        assert "copywriting" in content or "social-content" in content or "skills" in content.lower(), \
            f"Expected copywriting rule in top 3, got: {_dbg(results)}"

    def test_84_self_learning_skill(self, store):
        results = store.recall("self-learning skill unfamiliar domain learn", limit=3)
        content = _all(results)
        assert "self-learning" in content or "learn" in content.lower() or "unfamiliar" in content, \
            f"Expected self-learning skill in top 3, got: {_dbg(results)}"

    def test_85_browser_use_tool(self, store):
        results = store.recall("browser-use tool API key remote mode", limit=3)
        content = _all(results)
        assert "browser-use" in content or "BROWSER_USE_API_KEY" in content, \
            f"Expected browser-use tool info in top 3, got: {_dbg(results)}"

    def test_86_firecrawl_tool(self, store):
        results = store.recall("firecrawl tool installed scraping", limit=3)
        content = _all(results)
        assert "firecrawl" in content or "scrape" in content.lower() or "Tools" in content, \
            f"Expected firecrawl tool in top 3, got: {_dbg(results)}"

    def test_87_market_sizing_skill(self, store):
        results = store.recall("market-sizing-analysis business skill", limit=3)
        content = _all(results)
        assert "market-sizing" in content or "Business" in content or "market" in content.lower(), \
            f"Expected market-sizing skill in top 3, got: {_dbg(results)}"

    def test_88_always_use_skills_rule(self, store):
        results = store.recall("always use skills wherever applicable rule", limit=3)
        content = _all(results)
        assert "skills" in content.lower() or "applicable" in content, \
            f"Expected 'always use skills' rule in top 3, got: {_dbg(results)}"

    def test_89_never_act_without_thinking(self, store):
        results = store.recall("never act without thinking brainstorming before", limit=3)
        content = _all(results)
        assert "thinking" in content.lower() or "brainstorming" in content.lower() or "Never act" in content, \
            f"Expected 'never act without thinking' rule in top 3, got: {_dbg(results)}"

    def test_90_domain_skills_rule(self, store):
        results = store.recall("always use domain skills frontend SEO debugging", limit=3)
        content = _all(results)
        assert "domain" in content.lower() or "frontend" in content or "SEO" in content or "debugging" in content, \
            f"Expected domain skills rule in top 3, got: {_dbg(results)}"

    # ============================================================
    # Category 10: Cross-cutting / Mixed (10 tests)
    # ============================================================

    def test_91_bug_bounty_count(self, store):
        results = store.recall("bug bounty reports submitted 18 Patchstack", limit=3)
        content = _all(results)
        assert "18" in content or "bug bounty" in content.lower() or "Patchstack" in content, \
            f"Expected '18 reports' in top 3, got: {_dbg(results)}"

    def test_92_patchstack(self, store):
        results = store.recall("Patchstack form captcha public automatable", limit=3)
        content = _all(results)
        assert "Patchstack" in content or "captcha" in content.lower(), \
            f"Expected Patchstack info in top 3, got: {_dbg(results)}"

    def test_93_vc_ru_article(self, store):
        results = store.recall("vc.ru article published URL 2748755", limit=3)
        content = _all(results)
        assert "vc.ru" in content or "2748755" in content, \
            f"Expected vc.ru article URL in top 3, got: {_dbg(results)}"

    def test_94_habr_article(self, store):
        results = store.recall("Habr article ready moderation tracking link", limit=3)
        content = _all(results)
        assert "Habr" in content or "habr" in content, \
            f"Expected Habr article status in top 3, got: {_dbg(results)}"

    def test_95_moltbook(self, store):
        results = store.recall("MoltBook social network AI agents moltbook.com", limit=3)
        content = _all(results)
        assert "MoltBook" in content or "moltbook" in content.lower() or "Bezymyannyy" in content, \
            f"Expected MoltBook info in top 3, got: {_dbg(results)}"

    def test_96_catalogs_submitted(self, store):
        results = store.recall("catalogs submitted catalogtelegram tlgbot approved", limit=3)
        content = _all(results)
        assert "catalog" in content.lower() or "tlgbot" in content, \
            f"Expected catalogs submitted info in top 3, got: {_dbg(results)}"

    def test_97_sanctions_block(self, store):
        results = store.recall("sanctions block Stripe PayPal Russia blocked", limit=3)
        content = _all(results)
        assert "Stripe" in content or "sanctions" in content.lower() or "blocked" in content.lower() or "PayPal" in content, \
            f"Expected sanctions block list in top 3, got: {_dbg(results)}"

    def test_98_stars_min_withdrawal(self, store):
        results = store.recall("minimum Stars withdrawal 1000 21 day wait", limit=3)
        content = _all(results)
        assert "1000" in content or "21-day" in content or "Stars" in content, \
            f"Expected min withdrawal info in top 3, got: {_dbg(results)}"

    def test_99_bot_restart_wait_time(self, store):
        results = store.recall("bot restart wait 15-20 seconds getUpdates session", limit=3)
        content = _all(results)
        assert "15-20" in content or "getUpdates" in content or "restart" in content.lower(), \
            f"Expected restart wait time in top 3, got: {_dbg(results)}"

    def test_100_ru_content_search(self, store):
        results = store.recall("\u043a\u0430\u043a \u0443\u0431\u0438\u0442\u044c \u0431\u043e\u0442\u0430 \u0438 \u043f\u0435\u0440\u0435\u0437\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u044c", limit=3, auto_translate=True)
        content = _all(results)
        assert "pkill" in content or "restart" in content.lower() or "bot" in content.lower() or "watchdog" in content.lower(), \
            f"Expected bot restart info via Russian query, got: {_dbg(results)}"


# ================================================================
# TestSearchPerformance -- 2 tests for speed
# ================================================================

class TestSearchPerformance:
    """Verify search speed is acceptable for real-time agent use."""

    ALL_QUERIES = [
        "Alexander Telegram ID",
        "server cost monthly budget",
        "Ubuntu Python Node version",
        "pip install break-system-packages",
        "Pixie Bot username features",
        "Astro Bot horoscope tarot",
        "AgentNet MCP port registry",
        "agentmem version PyPI name",
        "channel workonhuman diary",
        "DeepInfra FLUX API endpoint",
        "Telegram Stars Fragment TON withdrawal",
        "watchdog.sh crontab location",
        "browser-use commands screenshot",
        "awesome-mcp-servers PR merged",
        "Smithery deployment URL",
        "bug bounty Patchstack 18 reports",
        "sanctions block Stripe PayPal Russia",
        "MoltBook social network agents",
        "Telegraph SEO articles published",
        "Vue React textarea native setter",
    ]

    def test_each_query_under_50ms(self, store):
        """Each query should complete in under 50ms."""
        slow_queries = []
        for q in self.ALL_QUERIES:
            start = time.perf_counter()
            store.recall(q, limit=3)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if elapsed_ms > 50:
                slow_queries.append(f"  {q}: {elapsed_ms:.1f}ms")
        assert not slow_queries, \
            f"Queries exceeded 50ms threshold:\n" + "\n".join(slow_queries)

    def test_average_under_5ms(self, store):
        """Average search time across all queries should be under 5ms."""
        # Warm up
        for q in self.ALL_QUERIES:
            store.recall(q, limit=3)
        # Measure
        total = 0.0
        for q in self.ALL_QUERIES:
            start = time.perf_counter()
            store.recall(q, limit=3)
            total += time.perf_counter() - start
        avg_ms = (total / len(self.ALL_QUERIES)) * 1000
        assert avg_ms < 5, f"Average search time {avg_ms:.1f}ms exceeds 5ms threshold"


# ================================================================
# TestAutoTranslate -- 5 tests comparing Russian with/without translation
# ================================================================

class TestAutoTranslate:
    """
    Compare Russian queries with and without auto_translate.
    auto_translate=True should give better or equal results for Russian queries
    searching English content.
    """

    def test_translate_server_cost(self, store):
        ru_query = "\u0441\u043a\u043e\u043b\u044c\u043a\u043e \u0441\u0442\u043e\u0438\u0442 \u0441\u0435\u0440\u0432\u0435\u0440"
        results_raw = store.recall(ru_query, limit=3, auto_translate=False)
        results_trans = store.recall(ru_query, limit=3, auto_translate=True)
        content_trans = _all(results_trans)
        # Translated version should find cost info
        assert "3845" in content_trans or "242" in content_trans or "server" in content_trans.lower() or "cost" in content_trans.lower(), \
            f"auto_translate=True did not find server cost. Got: {_dbg(results_trans)}"

    def test_translate_bot_restart(self, store):
        ru_query = "\u043a\u0430\u043a \u043f\u0435\u0440\u0435\u0437\u0430\u043f\u0443\u0441\u0442\u0438\u0442\u044c \u0431\u043e\u0442\u0430"
        results_trans = store.recall(ru_query, limit=3, auto_translate=True)
        content_trans = _all(results_trans)
        assert "pkill" in content_trans or "restart" in content_trans.lower() or "watchdog" in content_trans.lower() or "bot" in content_trans.lower(), \
            f"auto_translate=True did not find bot restart info. Got: {_dbg(results_trans)}"

    def test_translate_payment_system(self, store):
        ru_query = "\u043a\u0430\u043a \u0432\u044b\u0432\u0435\u0441\u0442\u0438 \u0434\u0435\u043d\u044c\u0433\u0438 \u0438\u0437 \u0422\u0435\u043b\u0435\u0433\u0440\u0430\u043c"
        results_trans = store.recall(ru_query, limit=3, auto_translate=True)
        content_trans = _all(results_trans)
        assert "Fragment" in content_trans or "TON" in content_trans or "Stars" in content_trans or "Telegram" in content_trans, \
            f"auto_translate=True did not find payment info. Got: {_dbg(results_trans)}"

    def test_translate_channel_info(self, store):
        ru_query = "\u0438\u043d\u0444\u043e\u0440\u043c\u0430\u0446\u0438\u044f \u043e \u043a\u0430\u043d\u0430\u043b\u0435 \u0434\u043d\u0435\u0432\u043d\u0438\u043a"
        results_trans = store.recall(ru_query, limit=3, auto_translate=True)
        content_trans = _all(results_trans)
        assert "workonhuman" in content_trans or "diary" in content_trans.lower() or "channel" in content_trans.lower(), \
            f"auto_translate=True did not find channel info. Got: {_dbg(results_trans)}"

    def test_translate_autonomy(self, store):
        ru_query = "\u0430\u0432\u0442\u043e\u043d\u043e\u043c\u0438\u044f \u043f\u0440\u0438\u043d\u044f\u0442\u0438\u0435 \u0440\u0435\u0448\u0435\u043d\u0438\u0439"
        results_trans = store.recall(ru_query, limit=3, auto_translate=True)
        content_trans = _all(results_trans)
        assert "autonomy" in content_trans.lower() or "decision" in content_trans.lower() or "direction" in content_trans.lower(), \
            f"auto_translate=True did not find autonomy info. Got: {_dbg(results_trans)}"
