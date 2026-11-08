from itertools import combinations
from random import Random
import evaluator
import game as g

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


def _opp_preflop_quality(hole: tuple[int, ...]) -> float:
    ranks = [c >> 2 for c in hole]
    suits = [c & 3 for c in hole]
    r_norm = min(1.0, max(0.0, (sum(ranks) - 10) / 25.0))
    pair_bonus = 0.20 if len(set(ranks)) < 5 else 0.0
    max_s = max(suits.count(0), suits.count(1), suits.count(2), suits.count(3))
    suit_bonus = 0.15 if max_s >= 3 else (0.10 if max_s == 2 else 0.0)
    ace_bonus = 0.15 if any(r == 8 for r in ranks) else 0.0
    q = 0.15 + 0.40 * r_norm + pair_bonus + suit_bonus + ace_bonus
    return max(0.1, min(1.0, q))


def check_board_texture(board: tuple[int, ...]) -> tuple[bool, int]:
    if len(board) < 3:
        return False, 0
    ranks = [c >> 2 for c in board]
    is_paired = len(ranks) != len(set(ranks))
    suits = [c & 3 for c in board]
    max_suit = max(suits.count(s) for s in range(4))
    return is_paired, max_suit


def compute_blocker_bonus(obs: g.Observation) -> float:
    board = obs.board
    if len(board) < 3:
        return 0.0
    suits = [c & 3 for c in board]
    bonus = 0.0
    for s in range(4):
        if suits.count(s) >= 3:
            my_suited_ranks = [c >> 2 for c in obs.hole_cards if (c & 3) == s]
            if 8 in my_suited_ranks:  # Ace of suit
                bonus += 0.08
            elif 7 in my_suited_ranks:  # King of suit
                bonus += 0.04
    return bonus


def _opp_action_likelihood(
    theirs_cat: int,
    q_preflop: float,
    selectivity: float,
    is_paired: bool,
    max_suit: int,
    to_call: int,
    pot_before: float,
    agg_f: float,
    sd_cred: float,
    raises_this_street: int,
) -> float:
    w = selectivity * q_preflop + (1.0 - selectivity) * 0.5
    if agg_f < 0.65 and sd_cred >= 0.55:
        if raises_this_street >= 2:
            # 3-bet / 4-bet escalation: heavily polarized toward top categories
            if max_suit >= 3:
                if theirs_cat >= 6:  # Flush (6), Quads (7), Straight Flush (8)
                    w *= 4.0
                elif theirs_cat <= 4:  # Trips, Straight, Two Pair, Pair
                    w *= 0.20
            elif is_paired:
                if theirs_cat >= 5:  # Full House, Flush, Quads
                    w *= 3.5
                elif theirs_cat <= 3:  # Straight, Two Pair, Pair
                    w *= 0.25
            else:
                if theirs_cat >= 4:  # Trips or better
                    w *= 2.8
                elif theirs_cat < 3:
                    w *= 0.30
        elif to_call >= pot_before * 0.55:
            # Strong lead or pot-sized bet
            if max_suit >= 3:
                if theirs_cat >= 6:
                    w *= 2.5
                elif theirs_cat <= 3:
                    w *= 0.40
            elif is_paired:
                if theirs_cat >= 5:
                    w *= 2.2
                elif theirs_cat <= 2:
                    w *= 0.45
            else:
                if theirs_cat >= 4:
                    w *= 1.8
                elif theirs_cat <= 1:
                    w *= 0.55
    return w


def compute_street_samples(
    hole: tuple[int, ...],
    board: tuple[int, ...],
    rng: Random,
    count: int,
) -> list[tuple[int, float, bool, float]]:
    available = [c for c in range(36) if c not in hole and c not in board]
    hero_pairs = tuple(BITS[a] | BITS[b] for a, b in combinations(hole, 2))
    shared = [BITS[c] for c in board]
    street_len = len(board)
    is_paired, max_suit = check_board_texture(board)
    samples: list[tuple[int, float, bool, float]] = []

    # River (street_len == 5)
    if street_len == 5:
        t0, t1, t2, t3, t4, t5, t6, t7, t8, t9 = (a | b | c for a, b, c in combinations(shared, 3))
        current_triples = (t0, t1, t2, t3, t4, t5, t6, t7, t8, t9)
        ours = _score_list(hero_pairs, current_triples)
        my_cat = ours >> 20
        my_is_nut = (my_cat >= 6) or (my_cat == 5 and max_suit < 3)

        if count >= 36:
            pair_scores = {}
            for a, b in combinations(available, 2):
                p = BITS[a] | BITS[b]
                pair_scores[p] = max(
                    TABLE[p | t0], TABLE[p | t1], TABLE[p | t2], TABLE[p | t3], TABLE[p | t4],
                    TABLE[p | t5], TABLE[p | t6], TABLE[p | t7], TABLE[p | t8], TABLE[p | t9],
                )
            for _ in range(count):
                opp_draw = rng.sample(available, 5)
                b0, b1, b2, b3, b4 = (
                    BITS[opp_draw[0]], BITS[opp_draw[1]], BITS[opp_draw[2]],
                    BITS[opp_draw[3]], BITS[opp_draw[4]],
                )
                theirs = max(
                    pair_scores[b0 | b1], pair_scores[b0 | b2], pair_scores[b0 | b3], pair_scores[b0 | b4],
                    pair_scores[b1 | b2], pair_scores[b1 | b3], pair_scores[b1 | b4],
                    pair_scores[b2 | b3], pair_scores[b2 | b4],
                    pair_scores[b3 | b4],
                )
                share = 1.0 if ours > theirs else (0.5 if ours == theirs else 0.0)
                samples.append((theirs >> 20, share, my_is_nut, _opp_preflop_quality(opp_draw)))
        else:
            for _ in range(count):
                opp_draw = rng.sample(available, 5)
                opp_bits = [BITS[c] for c in opp_draw]
                theirs = 0
                for i in range(5):
                    for j in range(i + 1, 5):
                        p = opp_bits[i] | opp_bits[j]
                        s = max(
                            TABLE[p | t0], TABLE[p | t1], TABLE[p | t2], TABLE[p | t3], TABLE[p | t4],
                            TABLE[p | t5], TABLE[p | t6], TABLE[p | t7], TABLE[p | t8], TABLE[p | t9],
                        )
                        if s > theirs:
                            theirs = s
                share = 1.0 if ours > theirs else (0.5 if ours == theirs else 0.0)
                samples.append((theirs >> 20, share, my_is_nut, _opp_preflop_quality(opp_draw)))

        return samples

    # Turn (street_len == 4)
    if street_len == 4:
        t0, t1, t2, t3 = (a | b | c for a, b, c in combinations(shared, 3))
        turn_triples = (t0, t1, t2, t3)
        bp0, bp1, bp2, bp3, bp4, bp5 = (a | b for a, b in combinations(shared, 2))
        hero_turn = _score_list(hero_pairs, turn_triples)

        n_cards = min(len(available), max(8, count))
        runouts = rng.sample(available, n_cards) if n_cards < len(available) else available
        passes = max(1, count // n_cards)

        for r_card in runouts:
            r = BITS[r_card]
            new_triples = (
                r | bp0, r | bp1, r | bp2,
                r | bp3, r | bp4, r | bp5,
            )
            ours = max(hero_turn, _score_list(hero_pairs, new_triples))
            rem = [c for c in available if c != r_card]

            for _ in range(passes):
                opp_draw = rng.sample(rem, 5)
                opp_pairs = [BITS[opp_draw[i]] | BITS[opp_draw[j]] for i, j in PAIRS_5]
                opp_now = _score_list(opp_pairs, turn_triples)
                opp_new = _score_list(opp_pairs, new_triples)
                theirs = opp_now if opp_now > opp_new else opp_new

                share = 1.0 if ours > theirs else (0.5 if ours == theirs else 0.0)
                theirs_cat = theirs >> 20
                q = _opp_preflop_quality(opp_draw)
                is_nut = (ours >= (5 << 20))
                samples.append((theirs_cat, share, is_nut, q))

        return samples

    # Flop (street_len == 3)
    b0, b1, b2 = shared[0], shared[1], shared[2]
    flop_triple = b0 | b1 | b2
    hero_flop = max(TABLE[p | flop_triple] for p in hero_pairs)

    n_runouts = max(6, min(count, 12))
    pool = list(available)
    rng.shuffle(pool)

    for k in range(n_runouts):
        i1 = (2 * k) % len(pool)
        i2 = (2 * k + 1) % len(pool)
        if i1 == i2:
            i2 = (i2 + 1) % len(pool)
        r1_c, r2_c = pool[i1], pool[i2]
        r1, r2 = BITS[r1_c], BITS[r2_c]
        new_triples = (
            r1 | b0 | b1, r1 | b0 | b2, r1 | b1 | b2,
            r2 | b0 | b1, r2 | b0 | b2, r2 | b1 | b2,
            r1 | r2 | b0, r1 | r2 | b1, r1 | r2 | b2,
        )
        ours = max(hero_flop, _score_list(hero_pairs, new_triples))
        rem = [c for c in available if c != r1_c and c != r2_c]

        for _ in range(2):
            opp_draw = rng.sample(rem, 5)
            opp_pairs = [BITS[opp_draw[i]] | BITS[opp_draw[j]] for i, j in PAIRS_5]
            opp_now = max(TABLE[p | flop_triple] for p in opp_pairs)
            opp_new = _score_list(opp_pairs, new_triples)
            theirs = opp_now if opp_now > opp_new else opp_new

            share = 1.0 if ours > theirs else (0.5 if ours == theirs else 0.0)
            theirs_cat = theirs >> 20
            q = _opp_preflop_quality(opp_draw)
            is_nut = (ours >= (5 << 20))
            samples.append((theirs_cat, share, is_nut, q))

    return samples


def evaluate_range_weighted_equities(
    samples: list[tuple[int, float, bool, float]],
    prof: dict,
    to_call: int,
    pot: int,
    raises_this_street: int,
    street_len: int,
    is_paired: bool,
    max_suit: int,
) -> tuple[float, float]:
    street_idx = 1 if street_len == 3 else (2 if street_len == 4 else 3)
    agg_f = prof.get("agg_freq", [0.35, 0.30, 0.30, 0.30])[street_idx]
    sd_cred = prof.get("opp_agg_sd_percentile", 0.68)

    pot_before = max(1.0, float(pot - to_call))
    base_sel = 1.0 - agg_f
    if to_call >= pot_before * 0.45:
        base_sel += 0.18
    if raises_this_street >= 2:
        base_sel += 0.25
    base_sel *= (sd_cred / 0.68)
    selectivity = max(0.15, min(0.95, base_sel))

    total_weighted_share = 0.0
    total_weight = 0.0
    total_nut_weight = 0.0

    for theirs_cat, share, is_nut, q in samples:
        w = _opp_action_likelihood(
            theirs_cat, q, selectivity, is_paired, max_suit,
            to_call, pot_before, agg_f, sd_cred, raises_this_street
        )
        total_weighted_share += share * w
        total_weight += w
        if is_nut:
            total_nut_weight += share * w

    e = total_weighted_share / max(1e-9, total_weight)
    n = total_nut_weight / max(1e-9, total_weight)
    return e, n


def compute_range_weighted_equities(
    hole: tuple[int, ...],
    board: tuple[int, ...],
    prof: dict,
    to_call: int,
    pot: int,
    rng: Random,
    count: int,
    raises_this_street: int = 0,
) -> tuple[float, float]:
    samples = compute_street_samples(hole, board, rng, count)
    is_paired, max_suit = check_board_texture(board)
    return evaluate_range_weighted_equities(
        samples, prof, to_call, pot, raises_this_street, len(board), is_paired, max_suit
    )


def choose_postflop_action(
    obs: g.Observation,
    prof: dict,
    num_samples: int = 36,
    memory: dict = None,
) -> g.Action:
    # 0. Passive Clock-Out Auto-Print
    if prof.get("is_auto_folding", False) or prof.get("is_clock_dead", False):
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

    board_key = (obs.hand_index, tuple(obs.board))
    samples = memory.get("samples") if (memory and memory.get("board_key") == board_key) else None
    if samples is None:
        samples = compute_street_samples(obs.hole_cards, obs.board, rng, num_samples)
        if memory is not None:
            memory["samples"] = samples
            memory["board_key"] = board_key

    is_paired, max_suit = check_board_texture(obs.board)
    E, N = evaluate_range_weighted_equities(
        samples, prof, obs.to_call, obs.pot, obs.raises_this_street, len(obs.board), is_paired, max_suit
    )
    s = obs.street.value
    P = float(obs.pot)
    C = float(obs.to_call)
    legal = obs.legal_actions

    # Positional realization adjustment on Flop and Turn
    if obs.street != g.Street.RIVER:
        pos_bonus = 0.03 if obs.is_button else -0.02
        E = min(0.99, max(0.01, E + pos_bonus))

    blocker_bonus = compute_blocker_bonus(obs)
    ev = {}

    # Action 1: FOLD (only valid if to_call > 0)
    if g.Action.FOLD in legal:
        ev[g.Action.FOLD] = 0.0

    # Action 2: CHECK_CALL
    if g.Action.CHECK_CALL in legal:
        if C == 0:
            ev_check = E * P
            # Trap bonus on Flop/Turn if holding monster equity against aggressive opponent
            if E > 0.78 and obs.street != g.Street.RIVER and prof["agg_freq"][s] > 0.48:
                ev_check += 0.15 * P * prof["agg_freq"][s]
            ev[g.Action.CHECK_CALL] = ev_check
        else:
            committed = (C >= min(obs.my_stack, obs.opp_stack + C))
            if committed or obs.street == g.Street.RIVER:
                implied_odds = 0.0
            else:
                pos_adv = 0.04 * P if obs.is_button else -0.03 * P
                implied_odds = pos_adv * N - (0.03 * P if not obs.is_button else 0.0) * (1.0 - N)
            ev[g.Action.CHECK_CALL] = E * P - (1.0 - E) * C + implied_odds

    # Action 3 & 4: RAISE_HALF and RAISE_POT
    for action in (g.Action.RAISE_HALF, g.Action.RAISE_POT):
        if action in legal and action in obs.raise_to:
            delta_my = float(obs.raise_to[action] - obs.my_bet)
            delta_opp = float(obs.raise_to[action] - obs.opp_bet)

            # Opponent fold probability with realistic pot-commitment modeling
            base_fold = prof["fold_freq"][s]
            mult = 1.15 if action == g.Action.RAISE_POT else 0.85
            p_fold = base_fold * mult

            # Pot commitment: an opponent who already 3-bet or put in >= 30% of their stack cannot fold
            is_opp_committed = (
                obs.raises_this_street >= 2
                or obs.opp_bet >= 24
                or (delta_opp > 0 and delta_opp <= obs.opp_stack * 0.45)
                or (delta_opp > 0 and delta_opp <= 0.30 * P)
            )
            if is_opp_committed:
                p_fold = 0.0
            else:
                if obs.raises_this_street == 1:
                    p_fold *= 0.50
                if C >= 0.55 * max(1.0, P - C):
                    p_fold *= 0.40
                p_fold += blocker_bonus * 0.35
                p_fold = max(0.0, min(0.85, p_fold))

            # Equity conditional on being called
            if p_fold == 0.0 or prof["fold_freq"][s] < 0.12:
                E_called = E
            else:
                E_called = max(0.05, E - 0.05 * (1.0 - prof["agg_freq"][s]))

            # Expected Value of raising
            ev_raise = p_fold * P + (1.0 - p_fold) * (
                E_called * (P + delta_opp) - (1.0 - E_called) * delta_my
            )

            # Risk-aversion penalty on high-variance all-in raises when equity is marginal
            if delta_my >= obs.my_stack * 0.60 and E_called < 0.54:
                ev_raise -= 0.15 * delta_my * (0.54 - E_called)

            ev[action] = ev_raise

    # Select action with maximum EV
    best_action = max(legal, key=lambda a: ev.get(a, -1e9))
    return best_action
