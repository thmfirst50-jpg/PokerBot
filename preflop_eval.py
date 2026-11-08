import numpy as np

# Precompute binomial coefficients C(n, k) for O(1) combinadic indexing of 5 sorted cards (0..35)
COMB = np.zeros((37, 6), dtype=np.int32)
for n in range(37):
    COMB[n, 0] = 1
    for k in range(1, min(n, 5) + 1):
        COMB[n, k] = COMB[n - 1, k - 1] + COMB[n - 1, k]


def hand_to_index(hole_cards) -> int:
    """
    Maps 5 sorted cards (0 <= c0 < c1 < c2 < c3 < c4 <= 35) to a unique integer in [0, 376991].
    obs.hole_cards is guaranteed sorted from low to high.
    """
    c0, c1, c2, c3, c4 = hole_cards
    return int(COMB[c0, 1] + COMB[c1, 2] + COMB[c2, 3] + COMB[c3, 4] + COMB[c4, 5])


def compute_preflop_percentile(hole_cards) -> float:
    """
    Deterministic, instant structural + nut-potential evaluator for Short Deck 5-Card PLO.
    Returns an estimated equity/strength score in [0.28, 0.76].
    Strictly accounts for:
      - 36-card ranks: 0=6, 1=7, ..., 7=K, 8=A (card >> 2)
      - Suits: 0..3 (card & 3)
      - Exact 2+3 Omaha rule (3+ cards of the same suit hurt flush outs!)
      - Short Deck rankings: Flush > Full House, Trips > Straight, Ace plays low in A-6-7-8-9
    """
    ranks = [c >> 2 for c in hole_cards]
    suits = [c & 3 for c in hole_cards]

    # 1. High card base value
    high_score = sum(ranks) / 40.0  # Max sum is 8+8+8+8+7 = 39

    # 2. Suit structure (Double-suited is optimal; 3+ of a suit penalizes self-blocked outs)
    suit_counts = [0, 0, 0, 0]
    suit_max_rank = [-1, -1, -1, -1]
    for r, s in zip(ranks, suits):
        suit_counts[s] += 1
        if r > suit_max_rank[s]:
            suit_max_rank[s] = r

    suit_bonus = 0.0
    for s in range(4):
        if suit_counts[s] == 2:
            # Nut flush potential bonus (Ace=8, King=7, Queen=6)
            suit_bonus += 0.045 + 0.012 * max(0, suit_max_rank[s] - 4)
        elif suit_counts[s] == 3:
            # Exact 2+3 rule: 3rd suited card blocks our own flush outs
            suit_bonus += 0.020 + 0.008 * max(0, suit_max_rank[s] - 4)
        elif suit_counts[s] >= 4:
            suit_bonus -= 0.015

    # 3. Pair / Set potential (Pairs convert to Full Houses and Quads; Trips > Straight)
    rank_counts = {}
    for r in ranks:
        rank_counts[r] = rank_counts.get(r, 0) + 1

    pair_bonus = 0.0
    pairs = []
    for r, count in rank_counts.items():
        if count == 2:
            pairs.append(r)
            pair_bonus += 0.035 + 0.009 * r
        elif count == 3:
            # Trips in hand blocks set outs on board under 2+3 rule
            pair_bonus += 0.025 + 0.005 * r
        elif count == 4:
            pair_bonus -= 0.020
    if len(pairs) == 2:
        pair_bonus += 0.035  # Double-paired hand

    # 4. Connectivity / Wrap potential (including Ace-low: Ace=8 also acts as -1 for A-6-7-8-9)
    unique_ranks = set(ranks)
    if 8 in unique_ranks:
        unique_ranks.add(-1)  # Ace plays low with 6(0), 7(1), 8(2), 9(3)

    max_window = 0
    for start in range(-1, 5):
        in_window = sum(1 for r in range(start, start + 5) if r in unique_ranks)
        if in_window > max_window:
            max_window = in_window

    wrap_bonus = 0.0
    if max_window >= 5:
        wrap_bonus = 0.085
    elif max_window == 4:
        wrap_bonus = 0.055
    elif max_window == 3:
        wrap_bonus = 0.020

    raw = 0.34 + 0.14 * high_score + suit_bonus + pair_bonus + wrap_bonus
    return float(min(0.76, max(0.28, raw)))
