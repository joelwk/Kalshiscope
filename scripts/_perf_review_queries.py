"""Custom SQL analysis for performance review (read-only)."""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from research_profiles import family_from_text  # noqa: E402

DB_PATH = Path("data/market_state.db")
SECTION_RULE = "=" * 78

TRADED_ACTIONS = ("order_attempt",)


def _print_section(title: str) -> None:
    print()
    print(SECTION_RULE)
    print(title)
    print(SECTION_RULE)


def _safe_json_loads(text: object) -> dict | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _family(market_id: str) -> str:
    return family_from_text(market_id)


def _tier(c: float) -> str:
    if c >= 0.90:
        return "0.90+"
    if c >= 0.80:
        return "0.80-0.89"
    if c >= 0.70:
        return "0.70-0.79"
    if c >= 0.60:
        return "0.60-0.69"
    return "<0.60"


def main() -> None:
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    _print_section("1. Win rate by confidence tier (trade_outcomes)")
    rows = conn.execute(
        """
        SELECT
            CASE
                WHEN confidence >= 0.90 THEN '0.90+'
                WHEN confidence >= 0.80 THEN '0.80-0.89'
                WHEN confidence >= 0.70 THEN '0.70-0.79'
                WHEN confidence >= 0.60 THEN '0.60-0.69'
                ELSE '<0.60'
            END AS tier,
            COUNT(*) AS n,
            SUM(CASE WHEN won = 1 THEN 1 ELSE 0 END) AS wins,
            ROUND(AVG(CASE WHEN won IS NOT NULL THEN won END), 4) AS win_rate,
            ROUND(SUM(pnl_estimate), 2) AS pnl,
            ROUND(SUM(amount_usdc), 2) AS deployed
        FROM trade_outcomes
        WHERE won IS NOT NULL
        GROUP BY tier
        ORDER BY tier DESC
        """
    ).fetchall()
    print(f"{'tier':<10} {'n':>6} {'wins':>6} {'win_rate':>10} {'pnl':>10} {'deployed':>11} {'roi':>8}")
    for row in rows:
        deployed = float(row["deployed"] or 0.0)
        pnl = float(row["pnl"] or 0.0)
        roi = (pnl / deployed * 100.0) if deployed else 0.0
        print(
            f"{row['tier']:<10} {row['n']:>6} {row['wins']:>6} "
            f"{(row['win_rate'] or 0):>10.4f} {pnl:>+10.2f} {deployed:>11.2f} {roi:>+7.2f}%"
        )

    _print_section("2. Overconfident losses (won=0, confidence>=0.85), worst 20 by pnl")
    rows = conn.execute(
        """
        SELECT market_id, confidence, implied_prob, pnl_estimate, amount_usdc, resolution_state
        FROM trade_outcomes
        WHERE won = 0 AND confidence >= 0.85
        ORDER BY pnl_estimate ASC
        LIMIT 20
        """
    ).fetchall()
    print(
        f"{'market_id':<42} {'conf':>5} {'impl':>5} {'pnl':>8} {'deployed':>9}"
    )
    for row in rows:
        market_id = (row["market_id"] or "")[:42]
        print(
            f"{market_id:<42} {float(row['confidence'] or 0):>5.2f} "
            f"{float(row['implied_prob'] or 0):>5.2f} "
            f"{float(row['pnl_estimate'] or 0):>+8.2f} {float(row['amount_usdc'] or 0):>9.2f}"
        )

    _print_section("2b. Overconfident loss families (won=0, confidence>=0.85)")
    rows = conn.execute(
        """
        SELECT market_id, confidence, pnl_estimate, amount_usdc
        FROM trade_outcomes
        WHERE won = 0 AND confidence >= 0.85
        """
    ).fetchall()
    fam_oc: dict[str, dict[str, float]] = defaultdict(
        lambda: {"n": 0, "pnl": 0.0, "deployed": 0.0}
    )
    for row in rows:
        fam = _family(str(row["market_id"] or ""))
        fam_oc[fam]["n"] += 1
        fam_oc[fam]["pnl"] += float(row["pnl_estimate"] or 0)
        fam_oc[fam]["deployed"] += float(row["amount_usdc"] or 0)
    print(f"{'family':<10} {'n':>4} {'pnl':>10} {'deployed':>10} {'avg_loss':>9}")
    for fam, stats in sorted(fam_oc.items(), key=lambda x: x[1]["pnl"]):
        n = int(stats["n"])
        pnl = stats["pnl"]
        deployed = stats["deployed"]
        avg = (pnl / n) if n else 0.0
        print(f"{fam:<10} {n:>4} {pnl:>+10.2f} {deployed:>10.2f} {avg:>+9.2f}")

    _print_section("3. PnL by edge_source (decision_receipts)")
    rows = conn.execute(
        """
        SELECT
            COALESCE(LOWER(json_extract(decision_json, '$.edge_source')), 'unknown') AS edge_source,
            COUNT(*) AS decisions,
            SUM(CASE WHEN final_action = 'order_attempt' THEN 1 ELSE 0 END) AS traded,
            ROUND(AVG(CAST(json_extract(audit_json, '$.pre_execution_final_score') AS REAL)), 4) AS avg_score
        FROM decision_receipts
        GROUP BY edge_source
        ORDER BY decisions DESC
        """
    ).fetchall()
    print(f"{'edge_source':<28} {'decisions':>9} {'traded':>7} {'avg_score':>10}")
    for row in rows:
        print(
            f"{(row['edge_source'] or '')[:28]:<28} {row['decisions']:>9} "
            f"{row['traded']:>7} {(row['avg_score'] or 0):>10.4f}"
        )

    _print_section(
        "3b. PnL by edge_source joined with trade_outcomes (resolved trades, final_action=order_attempt)"
    )
    rows = conn.execute(
        """
        SELECT
            COALESCE(LOWER(json_extract(dr.decision_json, '$.edge_source')), 'unknown') AS edge_source,
            COUNT(DISTINCT dr.market_id) AS resolved_markets,
            COUNT(*) AS resolved_rows,
            SUM(CASE WHEN tout.won = 1 THEN 1 ELSE 0 END) AS wins,
            ROUND(SUM(tout.pnl_estimate), 2) AS pnl,
            ROUND(SUM(tout.amount_usdc), 2) AS deployed,
            ROUND(AVG(tout.confidence), 4) AS avg_conf
        FROM decision_receipts dr
        JOIN trade_outcomes tout
          ON tout.market_id = dr.market_id
         AND tout.won IS NOT NULL
        WHERE dr.final_action = 'order_attempt'
        GROUP BY edge_source
        ORDER BY pnl ASC
        """
    ).fetchall()
    print(
        f"{'edge_source':<28} {'mkts':>5} {'rows':>5} {'wins':>5} {'pnl':>10} {'deployed':>10} {'roi':>8} {'avg_conf':>9}"
    )
    for row in rows:
        deployed = float(row["deployed"] or 0.0)
        pnl = float(row["pnl"] or 0.0)
        roi = (pnl / deployed * 100.0) if deployed else 0.0
        print(
            f"{(row['edge_source'] or '')[:28]:<28} {row['resolved_markets']:>5} "
            f"{row['resolved_rows']:>5} {row['wins']:>5} {pnl:>+10.2f} {deployed:>10.2f} {roi:>+7.2f}% "
            f"{(row['avg_conf'] or 0):>9.4f}"
        )

    _print_section(
        "3c. PnL by research_profile (audit_json.research_profile / decision_json.research_profile)"
    )
    rows = conn.execute(
        """
        SELECT
            COALESCE(
                LOWER(json_extract(dr.audit_json, '$.research_profile')),
                LOWER(json_extract(dr.decision_json, '$.research_profile')),
                'unknown'
            ) AS research_profile,
            COUNT(DISTINCT dr.market_id) AS resolved_markets,
            SUM(CASE WHEN tout.won = 1 THEN 1 ELSE 0 END) AS wins,
            ROUND(SUM(tout.pnl_estimate), 2) AS pnl,
            ROUND(SUM(tout.amount_usdc), 2) AS deployed,
            ROUND(AVG(tout.confidence), 4) AS avg_conf
        FROM decision_receipts dr
        JOIN trade_outcomes tout
          ON tout.market_id = dr.market_id
         AND tout.won IS NOT NULL
        WHERE dr.final_action = 'order_attempt'
        GROUP BY research_profile
        ORDER BY pnl ASC
        """
    ).fetchall()
    print(
        f"{'research_profile':<24} {'mkts':>5} {'wins':>5} {'pnl':>10} {'deployed':>10} {'roi':>8} {'avg_conf':>9}"
    )
    for row in rows:
        deployed = float(row["deployed"] or 0.0)
        pnl = float(row["pnl"] or 0.0)
        roi = (pnl / deployed * 100.0) if deployed else 0.0
        print(
            f"{(row['research_profile'] or '')[:24]:<24} {row['resolved_markets']:>5} "
            f"{row['wins']:>5} {pnl:>+10.2f} {deployed:>10.2f} {roi:>+7.2f}% "
            f"{(row['avg_conf'] or 0):>9.4f}"
        )

    _print_section(
        "4. Score-gate effectiveness — were blocked (should_trade=1, final_action=skip) trades correct?"
    )
    rows = conn.execute(
        """
        SELECT
            COALESCE(json_extract(audit_json, '$.market_family'), 'unknown') AS family,
            COUNT(*) AS blocked,
            ROUND(AVG(CAST(json_extract(decision_json, '$.confidence') AS REAL)), 4) AS avg_conf,
            ROUND(AVG(CAST(json_extract(audit_json, '$.pre_execution_final_score') AS REAL)), 4) AS avg_score
        FROM decision_receipts
        WHERE COALESCE(json_extract(decision_json, '$.should_trade'), 0) = 1
          AND final_action = 'skip'
        GROUP BY family
        ORDER BY blocked DESC
        """
    ).fetchall()
    print(f"{'family':<14} {'blocked':>7} {'avg_conf':>9} {'avg_score':>10}")
    for row in rows:
        print(
            f"{(row['family'] or '')[:14]:<14} {row['blocked']:>7} "
            f"{(row['avg_conf'] or 0):>9.4f} {(row['avg_score'] or 0):>10.4f}"
        )

    _print_section(
        "4b. Counterfactual win rate of skip-blocks that later resolved"
    )
    rows = conn.execute(
        """
        SELECT
            COALESCE(json_extract(dr.audit_json, '$.market_family'), 'unknown') AS family,
            COUNT(*) AS n_blocked_resolved,
            SUM(CASE WHEN tout.won = 1 THEN 1 ELSE 0 END) AS would_have_won,
            ROUND(AVG(tout.confidence), 4) AS avg_conf,
            ROUND(AVG(CAST(json_extract(dr.audit_json, '$.pre_execution_final_score') AS REAL)), 4) AS avg_score,
            ROUND(SUM(tout.pnl_estimate), 2) AS counterfactual_pnl,
            ROUND(SUM(tout.amount_usdc), 2) AS would_have_deployed
        FROM decision_receipts dr
        JOIN trade_outcomes tout
          ON tout.market_id = dr.market_id
         AND tout.won IS NOT NULL
        WHERE COALESCE(json_extract(dr.decision_json, '$.should_trade'), 0) = 1
          AND dr.final_action = 'skip'
        GROUP BY family
        ORDER BY n_blocked_resolved DESC
        """
    ).fetchall()
    print(
        f"{'family':<14} {'blocked_n':>9} {'would_win':>9} {'wr':>7} {'avg_conf':>9} {'avg_score':>10} {'cf_pnl':>9}"
    )
    for row in rows:
        n = int(row["n_blocked_resolved"] or 0)
        wins = int(row["would_have_won"] or 0)
        wr = (wins / n) if n else 0.0
        print(
            f"{(row['family'] or '')[:14]:<14} {n:>9} {wins:>9} {wr:>7.4f} "
            f"{(row['avg_conf'] or 0):>9.4f} {(row['avg_score'] or 0):>10.4f} "
            f"{float(row['counterfactual_pnl'] or 0):>+9.2f}"
        )

    _print_section(
        "4c. Counterfactual win rate by SPECIFIC skip reason (should_trade=1, final_reason ...)"
    )
    rows = conn.execute(
        """
        SELECT
            COALESCE(LOWER(dr.final_reason),
                     LOWER(json_extract(dr.audit_json,'$.final_reason')),
                     'unknown') AS reason,
            COUNT(*) AS n_blocked_resolved,
            SUM(CASE WHEN tout.won = 1 THEN 1 ELSE 0 END) AS would_have_won,
            ROUND(SUM(tout.pnl_estimate), 2) AS counterfactual_pnl,
            ROUND(AVG(tout.confidence), 4) AS avg_conf
        FROM decision_receipts dr
        JOIN trade_outcomes tout
          ON tout.market_id = dr.market_id
         AND tout.won IS NOT NULL
        WHERE COALESCE(json_extract(dr.decision_json, '$.should_trade'), 0) = 1
          AND dr.final_action = 'skip'
        GROUP BY reason
        ORDER BY n_blocked_resolved DESC
        """
    ).fetchall()
    print(f"{'reason':<32} {'n':>4} {'wins':>5} {'wr':>7} {'cf_pnl':>9} {'avg_conf':>9}")
    for row in rows:
        n = int(row["n_blocked_resolved"] or 0)
        wins = int(row["would_have_won"] or 0)
        wr = (wins / n) if n else 0.0
        print(
            f"{(row['reason'] or '')[:32]:<32} {n:>4} {wins:>5} {wr:>7.4f} "
            f"{float(row['counterfactual_pnl'] or 0):>+9.2f} {(row['avg_conf'] or 0):>9.4f}"
        )

    _print_section(
        "5. Recent settlement streak (last 50 settled trades) and longest losing/winning runs"
    )
    rows = conn.execute(
        """
        SELECT market_id, won, pnl_realized, contracts, settled_at
        FROM exchange_settlements
        ORDER BY settled_at DESC
        LIMIT 50
        """
    ).fetchall()
    print(f"{'market_id':<42} {'won':>3} {'pnl':>8} {'ctr':>4} settled_at")
    for row in rows:
        market_id = (row["market_id"] or "")[:42]
        won = int(row["won"] or 0) if row["won"] is not None else None
        pnl = float(row["pnl_realized"] or 0)
        contracts = int(row["contracts"] or 0)
        settled_at = row["settled_at"] or ""
        print(
            f"{market_id:<42} {(str(won) if won is not None else '?'):>3} "
            f"{pnl:>+8.2f} {contracts:>4} {settled_at}"
        )

    streak_rows = conn.execute(
        """
        SELECT won, pnl_realized, settled_at
        FROM exchange_settlements
        WHERE won IS NOT NULL AND contracts > 0
        ORDER BY settled_at ASC
        """
    ).fetchall()
    longest_loss = 0
    longest_win = 0
    cur_loss = 0
    cur_win = 0
    current_run_kind = None
    current_run_len = 0
    last_won = None
    for row in streak_rows:
        won = int(row["won"] or 0)
        if won == 1:
            cur_win += 1
            cur_loss = 0
            longest_win = max(longest_win, cur_win)
        else:
            cur_loss += 1
            cur_win = 0
            longest_loss = max(longest_loss, cur_loss)
        if last_won is None or last_won != won:
            current_run_kind = "win" if won == 1 else "loss"
            current_run_len = 1
        else:
            current_run_len += 1
        last_won = won
    print()
    print(f"longest_winning_streak={longest_win} settled trades")
    print(f"longest_losing_streak ={longest_loss} settled trades")
    print(f"current_streak={current_run_kind} length={current_run_len}")

    _print_section("6. API cost efficiency (cycle_receipts)")
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS cycles,
            SUM(CAST(json_extract(payload_json, '$.api_tokens_consumed') AS INTEGER)) AS total_tokens,
            ROUND(SUM(CAST(json_extract(payload_json, '$.api_cost_estimate_usd') AS REAL)), 4) AS total_cost,
            SUM(CAST(json_extract(payload_json, '$.order_attempts') AS INTEGER)) AS total_orders,
            SUM(CAST(json_extract(payload_json, '$.decisions_made') AS INTEGER)) AS total_decisions
        FROM cycle_receipts
        """
    ).fetchone()
    if row:
        cycles = int(row["cycles"] or 0)
        tokens = int(row["total_tokens"] or 0)
        cost = float(row["total_cost"] or 0)
        orders = int(row["total_orders"] or 0)
        decisions = int(row["total_decisions"] or 0)
        print(f"cycles={cycles}")
        print(f"total_tokens={tokens}")
        print(f"total_cost_usd=${cost:.4f}")
        print(f"total_order_attempts={orders}")
        print(f"total_decisions={decisions}")
        if orders:
            print(f"cost_per_order_attempt=${cost / orders:.4f}")
        if decisions:
            print(f"cost_per_decision=${cost / decisions:.4f}")
        if cycles:
            print(f"avg_cost_per_cycle=${cost / cycles:.4f}")

    _print_section("6b. Realized PnL vs API cost (lifetime)")
    realized = conn.execute(
        "SELECT ROUND(SUM(pnl_realized), 2) AS pnl, COUNT(*) AS n "
        "FROM exchange_settlements"
    ).fetchone()
    pnl = float(realized["pnl"] or 0)
    n = int(realized["n"] or 0)
    print(f"settled_trades={n}")
    print(f"realized_pnl=${pnl:.2f}")
    if row and float(row["total_cost"] or 0):
        print(f"net_pnl_after_api_cost=${pnl - float(row['total_cost']):.2f}")
    if n:
        print(f"avg_realized_pnl_per_settle=${pnl / n:.4f}")

    _print_section("6c. API cost per profitable trade")
    profitable = conn.execute(
        "SELECT COUNT(*) AS n FROM exchange_settlements WHERE pnl_realized > 0"
    ).fetchone()
    pnl_winners = conn.execute(
        "SELECT ROUND(SUM(pnl_realized),2) AS pnl FROM exchange_settlements WHERE pnl_realized > 0"
    ).fetchone()
    if profitable and row and float(row["total_cost"] or 0):
        n_profit = int(profitable["n"] or 0)
        gross_winnings = float(pnl_winners["pnl"] or 0)
        if n_profit:
            print(f"profitable_trades={n_profit}")
            print(f"total_winnings=${gross_winnings:.2f}")
            print(f"api_cost_per_profitable_trade=${float(row['total_cost']) / n_profit:.4f}")
            if gross_winnings:
                print(f"api_cost_to_gross_winnings_ratio={float(row['total_cost']) / gross_winnings:.4f}")

    _print_section("7. Daily PnL trend (settlements)")
    rows = conn.execute(
        """
        SELECT
            substr(settled_at, 1, 10) AS day,
            COUNT(*) AS n,
            SUM(CASE WHEN won = 1 THEN 1 ELSE 0 END) AS wins,
            ROUND(SUM(pnl_realized), 2) AS pnl
        FROM exchange_settlements
        WHERE won IS NOT NULL AND contracts > 0
        GROUP BY day
        ORDER BY day DESC
        LIMIT 30
        """
    ).fetchall()
    print(f"{'day':<12} {'n':>4} {'wins':>5} {'wr':>7} {'pnl':>10}")
    for row in rows:
        n = int(row["n"] or 0)
        wins = int(row["wins"] or 0)
        wr = (wins / n) if n else 0.0
        print(
            f"{(row['day'] or ''):<12} {n:>4} {wins:>5} {wr:>7.4f} "
            f"{float(row['pnl'] or 0):>+10.2f}"
        )

    _print_section("8. Confidence × family grid (n / wr / pnl)")
    rows = conn.execute(
        """
        SELECT market_id, confidence, won, pnl_estimate, amount_usdc
        FROM trade_outcomes
        WHERE won IS NOT NULL AND confidence IS NOT NULL
        """
    ).fetchall()
    grid: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"n": 0, "wins": 0, "pnl": 0.0, "deployed": 0.0}
    )
    for row in rows:
        fam = _family(str(row["market_id"] or ""))
        tier = _tier(float(row["confidence"] or 0))
        cell = grid[(fam, tier)]
        cell["n"] += 1
        cell["wins"] += 1 if int(row["won"] or 0) == 1 else 0
        cell["pnl"] += float(row["pnl_estimate"] or 0)
        cell["deployed"] += float(row["amount_usdc"] or 0)

    families = sorted({fam for (fam, _) in grid})
    tiers = ["0.90+", "0.80-0.89", "0.70-0.79", "0.60-0.69", "<0.60"]
    print(
        f"{'family':<10} | "
        + " | ".join(f"{t:^26}" for t in tiers)
    )
    for fam in families:
        parts = []
        for tier in tiers:
            cell = grid.get((fam, tier))
            if not cell or cell["n"] == 0:
                parts.append(f"{'-':^26}")
            else:
                wr = cell["wins"] / cell["n"]
                parts.append(
                    f"n={int(cell['n']):>3} wr={wr:>6.2%} pnl={cell['pnl']:>+6.2f}"
                )
        print(f"{fam:<10} | " + " | ".join(parts))

    _print_section("9. Decision_receipts top edge_source × profile counts")
    rows = conn.execute(
        """
        SELECT
            COALESCE(LOWER(json_extract(decision_json, '$.edge_source')), 'unknown') AS edge_source,
            COALESCE(LOWER(json_extract(audit_json, '$.research_profile')),
                     LOWER(json_extract(decision_json, '$.research_profile')),
                     'unknown') AS profile,
            COUNT(*) AS n,
            SUM(CASE WHEN final_action = 'order_attempt' THEN 1 ELSE 0 END) AS submitted,
            SUM(CASE WHEN final_action = 'skip' THEN 1 ELSE 0 END) AS skipped,
            SUM(CASE WHEN final_action = 'research_queued' THEN 1 ELSE 0 END) AS researched
        FROM decision_receipts
        GROUP BY edge_source, profile
        ORDER BY submitted DESC
        LIMIT 25
        """
    ).fetchall()
    print(
        f"{'edge_source':<18} {'profile':<22} {'n':>6} {'submit':>6} {'skip':>6} {'rsrch':>6}"
    )
    for row in rows:
        print(
            f"{(row['edge_source'] or '')[:18]:<18} "
            f"{(row['profile'] or '')[:22]:<22} "
            f"{row['n']:>6} {row['submitted']:>6} {row['skipped']:>6} {row['researched']:>6}"
        )

    _print_section("10. Calibration: confidence vs actual win-rate (Brier and miscalibration)")
    sample_rows = conn.execute(
        "SELECT confidence, won FROM trade_outcomes WHERE won IS NOT NULL AND confidence IS NOT NULL"
    ).fetchall()
    if sample_rows:
        confs = [float(r["confidence"]) for r in sample_rows]
        wons = [int(r["won"]) for r in sample_rows]
        n = len(confs)
        mean_c = sum(confs) / n
        mean_w = sum(wons) / n
        num = sum((c - mean_c) * (w - mean_w) for c, w in zip(confs, wons))
        den_c = sum((c - mean_c) ** 2 for c in confs) ** 0.5
        den_w = sum((w - mean_w) ** 2 for w in wons) ** 0.5
        corr = num / (den_c * den_w) if den_c > 0 and den_w > 0 else 0.0
        brier = sum((c - w) ** 2 for c, w in zip(confs, wons)) / n
        print(f"n={n}")
        print(f"avg_confidence={mean_c:.4f}")
        print(f"win_rate={mean_w:.4f}")
        print(f"pearson_r(confidence, won)={corr:.4f}")
        print(f"brier_score={brier:.4f}")
        bins = defaultdict(lambda: [0, 0])
        for c, w in zip(confs, wons):
            tier = _tier(c)
            bins[tier][0] += 1
            bins[tier][1] += w
        print()
        print(f"{'tier':<10} {'n':>4} {'avg_conf':>9} {'actual_wr':>10} {'gap':>7}")
        for tier in ["0.90+", "0.80-0.89", "0.70-0.79", "0.60-0.69", "<0.60"]:
            n_t, w_t = bins.get(tier, [0, 0])
            wr = (w_t / n_t) if n_t else 0.0
            tier_floors = {"0.90+": 0.95, "0.80-0.89": 0.85, "0.70-0.79": 0.75, "0.60-0.69": 0.65, "<0.60": 0.55}
            target = tier_floors[tier]
            gap = wr - target
            print(f"{tier:<10} {n_t:>4} {target:>9.2f} {wr:>10.4f} {gap:>+7.4f}")

    _print_section("11. Decision-receipt skip-reason breakdown (final_reason)")
    rows = conn.execute(
        """
        SELECT
            COALESCE(LOWER(final_reason),
                     LOWER(json_extract(audit_json, '$.final_reason')),
                     'unknown') AS reason,
            COUNT(*) AS n
        FROM decision_receipts
        WHERE final_action = 'skip'
        GROUP BY reason
        ORDER BY n DESC
        LIMIT 25
        """
    ).fetchall()
    for row in rows:
        print(f"  {row['reason']}: n={row['n']}")

    _print_section("11b. Skip reasons for should_trade=1 only")
    rows = conn.execute(
        """
        SELECT
            COALESCE(LOWER(final_reason),
                     LOWER(json_extract(audit_json, '$.final_reason')),
                     'unknown') AS reason,
            COUNT(*) AS n
        FROM decision_receipts
        WHERE final_action = 'skip'
          AND COALESCE(json_extract(decision_json, '$.should_trade'), 0) = 1
        GROUP BY reason
        ORDER BY n DESC
        LIMIT 25
        """
    ).fetchall()
    for row in rows:
        print(f"  {row['reason']}: n={row['n']}")

    _print_section("12. Latest 5 cycle_receipts payload summary")
    rows = conn.execute(
        """
        SELECT timestamp, payload_json
        FROM cycle_receipts
        ORDER BY timestamp DESC
        LIMIT 5
        """
    ).fetchall()
    for row in rows:
        payload = _safe_json_loads(row["payload_json"]) or {}
        keys = (
            "decisions_made",
            "order_attempts",
            "rejection_breakdown",
            "evidence_basis_breakdown",
            "api_tokens_consumed",
            "api_cost_estimate_usd",
        )
        slim = {k: payload.get(k) for k in keys}
        print(f"{row['timestamp']}: {json.dumps(slim, default=str)[:500]}")

    _print_section("13. Aggregate rejection_breakdown across all cycles")
    rows = conn.execute(
        """
        SELECT payload_json FROM cycle_receipts
        """
    ).fetchall()
    reject_total: Counter = Counter()
    evidence_total: Counter = Counter()
    for row in rows:
        payload = _safe_json_loads(row["payload_json"]) or {}
        rb = payload.get("rejection_breakdown") or {}
        if isinstance(rb, dict):
            for k, v in rb.items():
                try:
                    reject_total[k] += int(v)
                except (TypeError, ValueError):
                    continue
        eb = payload.get("evidence_basis_breakdown") or {}
        if isinstance(eb, dict):
            for k, v in eb.items():
                try:
                    evidence_total[k] += int(v)
                except (TypeError, ValueError):
                    continue
    print("Rejection breakdown (all cycles):")
    for k, v in sorted(reject_total.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print()
    print("Evidence basis breakdown (all cycles):")
    for k, v in sorted(evidence_total.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    _print_section("14. Skip-reason × outcome (resolved)")
    rows = conn.execute(
        """
        SELECT
            COALESCE(LOWER(dr.final_reason), 'unknown') AS reason,
            COUNT(*) AS n_resolved,
            SUM(CASE WHEN tout.won = 1 THEN 1 ELSE 0 END) AS would_win,
            SUM(CASE WHEN tout.won = 0 THEN 1 ELSE 0 END) AS would_lose,
            ROUND(SUM(tout.pnl_estimate), 2) AS cf_pnl
        FROM decision_receipts dr
        JOIN trade_outcomes tout
          ON tout.market_id = dr.market_id
         AND tout.won IS NOT NULL
        WHERE dr.final_action = 'skip'
          AND COALESCE(json_extract(dr.decision_json, '$.should_trade'), 0) = 1
        GROUP BY reason
        ORDER BY n_resolved DESC
        """
    ).fetchall()
    print(f"{'reason':<32} {'n':>4} {'wins':>5} {'loss':>5} {'wr':>7} {'cf_pnl':>9}")
    for row in rows:
        n = int(row["n_resolved"] or 0)
        wins = int(row["would_win"] or 0)
        losses = int(row["would_lose"] or 0)
        wr = (wins / n) if n else 0.0
        print(
            f"{(row['reason'] or '')[:32]:<32} {n:>4} {wins:>5} {losses:>5} {wr:>7.4f} "
            f"{float(row['cf_pnl'] or 0):>+9.2f}"
        )

    _print_section("15. Order-attempt outcome distribution (final_reason for order_attempt)")
    rows = conn.execute(
        """
        SELECT COALESCE(LOWER(final_reason), 'unknown') AS reason, COUNT(*) AS n
        FROM decision_receipts
        WHERE final_action = 'order_attempt'
        GROUP BY reason
        ORDER BY n DESC
        LIMIT 20
        """
    ).fetchall()
    for row in rows:
        print(f"  {row['reason']}: n={row['n']}")

    _print_section("16. Decision_receipts sample fields (one order_attempt row)")
    sample = conn.execute(
        "SELECT decision_json, audit_json, score_json, order_json FROM decision_receipts WHERE final_action='order_attempt' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if sample:
        for key in ("decision_json", "audit_json", "score_json", "order_json"):
            data = _safe_json_loads(sample[key]) or {}
            print(f"\n{key} keys:")
            for k in sorted(data.keys()):
                v_repr = json.dumps(data[k], default=str)[:120]
                print(f"  {k}: {v_repr}")

    _print_section("17. order_attempt rows joined to settlements: were they filled and what was PnL?")
    rows = conn.execute(
        """
        SELECT
            dr.market_id,
            COALESCE(LOWER(json_extract(dr.decision_json, '$.edge_source')), 'unknown') AS edge_source,
            COALESCE(LOWER(json_extract(dr.audit_json, '$.research_profile')),
                     LOWER(json_extract(dr.decision_json, '$.research_profile')), 'unknown') AS profile,
            CAST(json_extract(dr.decision_json, '$.confidence') AS REAL) AS confidence,
            es.won,
            es.pnl_realized,
            es.contracts,
            es.settled_at
        FROM decision_receipts dr
        LEFT JOIN exchange_settlements es ON es.market_id = dr.market_id
        WHERE dr.final_action = 'order_attempt'
          AND es.contracts > 0
        ORDER BY es.settled_at DESC
        LIMIT 30
        """
    ).fetchall()
    print(f"{'market':<40} {'edge':<10} {'prof':<14} {'conf':>5} {'won':>3} {'pnl':>7} {'ctr':>4}")
    for row in rows:
        print(
            f"{(row['market_id'] or '')[:40]:<40} "
            f"{(row['edge_source'] or '')[:10]:<10} "
            f"{(row['profile'] or '')[:14]:<14} "
            f"{float(row['confidence'] or 0):>5.2f} "
            f"{(str(int(row['won'] or 0)) if row['won'] is not None else '?'):>3} "
            f"{float(row['pnl_realized'] or 0):>+7.2f} "
            f"{int(row['contracts'] or 0):>4}"
        )

    conn.close()


if __name__ == "__main__":
    main()
