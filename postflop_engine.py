from itertools import combinations
from random import Random
import game as g
import evaluator

TABLE = evaluator.get_table()
BITS = tuple(1 << card for card in range(36))
PAIRS_5 = tuple(combinations(range(5), 2))


def _score_list(pairs, triples) -> int:
    best = 0
    for p in pairs:
        for t in triples:
            v = TABLE[p | t]
            if v > best:
                best = v
    return best


def compute_stratified_equities(hole, board, rng: Random, count: int):
    available = [BITS[c] for c in range(36) if c not in hole and c not in board]
    hero_pairs = tuple(BITS[a] | BITS[b] for a, b in combinations(hole, 2))
    shared = [BITS[c] for c in board]
    street_len = len(board)

    # River
    if street_len == 5:
        t0, t1, t2, t3, t4, t5, t6, t7, t8, t9 = (a | b | c for a, b, c in combinations(shared, 3))
        current_triples = (t0, t1, t2, t3, t4, t5, t6, t7, t8, t9)
        ours = _score_list(hero_pairs, current_triples)

        pair_scores = {}
        max_opp_possible = 0
        for a, b in combinations(available, 2):
            p = a | b
            s = max(
                TABLE[p | t0], TABLE[p | t1], TABLE[p | t2], TABLE[p | t3], TABLE[p | t4],
                TABLE[p | t5], TABLE[p | t6], TABLE[p | t7], TABLE[p | t8], TABLE[p | t9],
            )
            pair_scores[p] = s
            if s > max_opp_possible:
                max_opp_possible = s

        samples = []
        total_share = 0.0
        for _ in range(count):
            d0, d1, d2, d3, d4 = rng.sample(available, 5)
            theirs = max(
                pair_scores[d0 | d1], pair_scores[d0 | d2], pair_scores[d0 | d3], pair_scores[d0 | d4],
                pair_scores[d1 | d2], pair_scores[d1 | d3], pair_scores[d1 | d4],
                pair_scores[d2 | d3], pair_scores[d2 | d4],
                pair_scores[d3 | d4],
            )
            share = 1.0 if ours > theirs else (0.5 if ours == theirs else 0.0)
            total_share += share
            samples.append((theirs, share))

        samples.sort(key=lambda x: x[0])
        sorted_shares = [s for _, s in samples]
        raw_eq = total_share / count
        nut_eq = 1.0 if ours >= max_opp_possible else (raw_eq if (ours >> 20) >= 6 else 0.0)
        return raw_eq, nut_eq, sorted_shares

    # Turn
    if street_len == 4:
        t0, t1, t2, t3 = (a | b | c for a, b, c in combinations(shared, 3))
        turn_triples = (t0, t1, t2, t3)
        bp0, bp1, bp2, bp3, bp4, bp5 = (a | b for a, b in combinations(shared, 2))
        hero_turn = _score_list(hero_pairs, turn_triples)

        n_avail = len(available)
        passes = max(1, round(count / n_avail))
        samples = []
        total_share = 0.0
        nut_wins = 0.0

        for idx, r in enumerate(available):
            new_triples = (r | bp0, r | bp1, r | bp2, r | bp3, r | bp4, r | bp5)
            ours = max(hero_turn, _score_list(hero_pairs, new_triples))
            rem = available[:idx] + available[idx + 1:]

            for _ in range(passes):
                draw = rng.sample(rem, 5)
                opp_pairs = [draw[i] | draw[j] for i, j in PAIRS_5]
                opp_now = _score_list(opp_pairs, turn_triples)
                opp_new = _score_list(opp_pairs, new_triples)
                theirs = opp_now if opp_now > opp_new else opp_new

                share = 1.0 if ours > theirs else (0.5 if ours == theirs else 0.0)
                total_share += share
                if ours >= (6 << 20):
                    nut_wins += share
                opp_composite = (opp_now * 3) + (theirs * 2)
                samples.append((opp_composite, share))

        total_n = len(samples)
        samples.sort(key=lambda x: x[0])
        sorted_shares = [s for _, s in samples]
        return total_share / total_n, nut_wins / total_n, sorted_shares

    # Flop
    b0, b1, b2 = shared[0], shared[1], shared[2]
    flop_triple = b0 | b1 | b2
    hero_flop = max(TABLE[p | flop_triple] for p in hero_pairs)

    n_runouts = max(8, count // 2)
    samples = []
    total_share = 0.0
    nut_wins = 0.0

    pool = list(available)
    rng.shuffle(pool)
    for k in range(n_runouts):
        i1 = (2 * k) % len(pool)
        i2 = (2 * k + 1) % len(pool)
        if i1 == i2:
            i2 = (i2 + 1) % len(pool)
        r1, r2 = pool[i1], pool[i2]
        new_triples = (
            r1 | b0 | b1, r1 | b0 | b2, r1 | b1 | b2,
            r2 | b0 | b1, r2 | b0 | b2, r2 | b1 | b2,
            r1 | r2 | b0, r1 | r2 | b1, r1 | r2 | b2,
        )
        ours = max(hero_flop, _score_list(hero_pairs, new_triples))
        rem = [c for c in available if c != r1 and c != r2]

        for _ in range(2):
            draw = rng.sample(rem, 5)
            opp_pairs = [draw[i] | draw[j] for i, j in PAIRS_5]
            opp_now = max(TABLE[p | flop_triple] for p in opp_pairs)
            opp_new = _score_list(opp_pairs, new_triples)
            theirs = opp_now if opp_now > opp_new else opp_new

            share = 1.0 if ours > theirs else (0.5 if ours == theirs else 0.0)
            total_share += share
            if ours >= (6 << 20):
                nut_wins += share
            opp_composite = (opp_now * 3) + (theirs * 2)
            samples.append((opp_composite, share))

    total_n = len(samples)
    samples.sort(key=lambda x: x[0])
    sorted_shares = [s for _, s in samples]
    return total_share / total_n, nut_wins / total_n, sorted_shares


def _slice_equity(sorted_shares: list[float], low_frac: float, high_frac: float, noise_weight: float = 0.08) -> float:
    n = len(sorted_shares)
    if n == 0:
        return 0.5
    lo = max(0, min(n - 1, int(low_frac * n)))
    hi = max(lo + 1, min(n, int(high_frac * n)))
    in_sum = sum(sorted_shares[lo:hi])
    in_cnt = hi - lo
    out_sum = sum(sorted_shares[:lo]) + sum(sorted_shares[hi:])
    out_cnt = n - in_cnt
    num = in_sum + noise_weight * out_sum
    den = in_cnt + noise_weight * out_cnt
    return num / den if den > 0 else 0.5


def choose_postflop_action(obs: g.Observation, prof: dict, num_samples: int = 32, memory: dict = None) -> g.Action:
    # 0. Dead-Clock Exploit
    if prof.get("is_clock_dead", False):
        if g.Action.RAISE_HALF in obs.legal_actions:
            return g.Action.RAISE_HALF
        if g.Action.RAISE_POT in obs.legal_actions:
            return g.Action.RAISE_POT
        if g.Action.CHECK_CALL in obs.legal_actions:
            return g.Action.CHECK_CALL
        return obs.legal_actions[0]

    rng = memory.get("rng") if memory else None
    if rng is None:
        rng = Random(202605 + obs.hand_index)

    board_key = tuple(obs.board)
    eq_data = memory.get("eq_data") if (memory and memory.get("board_key") == board_key) else None
    if eq_data is None:
        s_len = len(obs.board)
        is_big = (obs.pot >= 16)
        if s_len == 5:
            count = 40 if is_big else 26
        elif s_len == 4:
            count = 27  # Exactly 27 river cards: full runout coverage
        else:
            count = 20 if is_big else 14
        eq_data = compute_stratified_equities(obs.hole_cards, obs.board, rng, count)
        if memory is not None:
            memory["eq_data"] = eq_data
            memory["board_key"] = board_key

    raw_eq, nut_eq, sorted_shares = eq_data
    river = (len(obs.board) == 5)
    turn = (len(obs.board) == 4)
    call_cost = obs.to_call
    total_pot = obs.pot + call_cost
    price = call_cost / total_pot if total_pot > 0 else 0.0
    legal = obs.legal_actions

    # 1. Blind Spammer Exploit
    if prof.get("is_blind_spammer", False):
        if call_cost == 0:
            if raw_eq >= 0.55:
                if g.Action.RAISE_POT in legal and raw_eq >= 0.63:
                    return g.Action.RAISE_POT
                if g.Action.RAISE_HALF in legal:
                    return g.Action.RAISE_HALF
            return g.Action.CHECK_CALL
        else:
            if raw_eq >= 0.58 or (raw_eq >= 0.52 and nut_eq >= 0.35):
                if g.Action.RAISE_POT in legal and raw_eq >= 0.63:
                    return g.Action.RAISE_POT
                if g.Action.RAISE_HALF in legal:
                    return g.Action.RAISE_HALF
            if raw_eq >= price:
                return g.Action.CHECK_CALL
            if g.Action.FOLD in legal:
                return g.Action.FOLD
            return g.Action.CHECK_CALL

    # 2. Case 1: Checked to us (call_cost == 0)
    if call_cost == 0:
        if obs.is_button:
            eq_vs_check = _slice_equity(sorted_shares, 0.0, 0.72, noise_weight=0.12)
            if river:
                if g.Action.RAISE_POT in legal and (raw_eq >= 0.655 or eq_vs_check >= 0.80):
                    return g.Action.RAISE_POT
                if g.Action.RAISE_HALF in legal and (raw_eq >= 0.515 or eq_vs_check >= 0.66):
                    return g.Action.RAISE_HALF
                return g.Action.CHECK_CALL
            else:
                if g.Action.RAISE_POT in legal and (raw_eq >= 0.675 or (raw_eq >= 0.62 and nut_eq >= 0.38)):
                    return g.Action.RAISE_POT
                if g.Action.RAISE_HALF in legal and (raw_eq >= 0.525 or eq_vs_check >= 0.65):
                    return g.Action.RAISE_HALF
                return g.Action.CHECK_CALL
        else:
            station_adj = -0.04 if prof.get("is_station", False) else 0.0
            if river:
                if g.Action.RAISE_POT in legal and raw_eq >= 0.72 + station_adj:
                    return g.Action.RAISE_POT
                if g.Action.RAISE_HALF in legal and raw_eq >= 0.565 + station_adj:
                    return g.Action.RAISE_HALF
                return g.Action.CHECK_CALL
            else:
                if g.Action.RAISE_POT in legal and raw_eq >= 0.71 + station_adj:
                    return g.Action.RAISE_POT
                if g.Action.RAISE_HALF in legal and raw_eq >= 0.575 + station_adj:
                    return g.Action.RAISE_HALF
                return g.Action.CHECK_CALL

    # 3. Case 2: Facing bet or raise (call_cost > 0)
    opp_is_pot = call_cost >= (obs.pot - call_cost) * 0.75
    opp_raised_us = obs.raises_this_street >= 2 or obs.my_bet > 0

    if opp_raised_us:
        low_cutoff = 0.75
        noise = 0.06
    elif opp_is_pot:
        low_cutoff = 0.62
        noise = 0.08
    else:
        low_cutoff = 0.50
        noise = 0.12

    if not river:
        noise = min(0.35, noise + (0.10 if not turn else 0.05))

    effective_eq = _slice_equity(sorted_shares, low_cutoff, 1.0, noise_weight=noise)

    # Raise for value
    if g.Action.RAISE_HALF in legal:
        raise_h = 0.64 + 0.04 * obs.raises_this_street
        raise_p = 0.74 + 0.04 * obs.raises_this_street
        if g.Action.RAISE_POT in legal and (effective_eq >= raise_p or (river and raw_eq >= 0.88)):
            return g.Action.RAISE_POT
        if effective_eq >= raise_h or (not river and raw_eq >= 0.78 and nut_eq >= 0.45):
            return g.Action.RAISE_HALF

    # Call vs Fold
    if g.Action.FOLD in legal:
        committed = call_cost >= min(obs.my_stack, obs.opp_stack + call_cost)
        pos_penalty = 0.025 if (not obs.is_button and not river) else 0.0
        implied_adj = -0.030 * nut_eq + 0.020 * (1.0 - nut_eq) if not river else 0.0
        future_cost = 0.0 if (committed or river) else max(0.0, 0.020 + pos_penalty + implied_adj)
        req_eq = price + future_cost
        if river and prof.get("post_agg", 0.30) < 0.35:
            req_eq += 0.030 if opp_is_pot else 0.015
        if effective_eq < req_eq:
            return g.Action.FOLD

    return g.Action.CHECK_CALL
