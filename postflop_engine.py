import random
from game import Action, Street, Observation
from evaluator import omaha_score
from preflop_eval import compute_preflop_percentile

ALL_CARDS = list(range(36))


def estimate_postflop_equity(obs: Observation, num_samples: int, opp_tightness: float) -> tuple[float, float]:
    """
    Returns (equity, nut_redraw_rate) in [0.0, 1.0].
    Uses rejection sampling on opponent hole cards when opponent has raised heavily.
    nut_redraw_rate measures how often our hand finishes as Flush (cat >= 6) or Full House (cat >= 5).
    """
    hole = list(obs.hole_cards)
    board = list(obs.board)
    known = set(hole) | set(board)
    deck = [c for c in ALL_CARDS if c not in known]

    cards_needed_board = 5 - len(board)
    wins = 0.0
    nut_finishes = 0.0

    # River optimization: our score is fixed, only sample opponent 5-card hands
    if cards_needed_board == 0:
        my_score = omaha_score(hole, board)
        my_cat = my_score >> 20
        is_nut_now = 1.0 if my_cat >= 6 else (0.6 if my_cat == 5 else 0.0)

        for _ in range(num_samples):
            opp_hole = random.sample(deck, 5)
            if opp_tightness > 0.45 and random.random() < 0.65:
                # Bias opponent toward stronger preflop hands when they raised
                if compute_preflop_percentile(sorted(opp_hole)) < opp_tightness:
                    opp_hole = random.sample(deck, 5)
            opp_score = omaha_score(opp_hole, board)
            if my_score > opp_score:
                wins += 1.0
            elif my_score == opp_score:
                wins += 0.5
        return wins / max(1, num_samples), is_nut_now

    # Flop / Turn rollout
    sample_count = 5 + cards_needed_board
    for _ in range(num_samples):
        drawn = random.sample(deck, sample_count)
        opp_hole = drawn[:5]
        if opp_tightness > 0.45 and random.random() < 0.55:
            if compute_preflop_percentile(sorted(opp_hole)) < opp_tightness:
                drawn = random.sample(deck, sample_count)
                opp_hole = drawn[:5]

        full_board = board + drawn[5:]
        my_score = omaha_score(hole, full_board)
        opp_score = omaha_score(opp_hole, full_board)

        if my_score > opp_score:
            wins += 1.0
        elif my_score == opp_score:
            wins += 0.5

        if (my_score >> 20) >= 6:
            nut_finishes += 1.0
        elif (my_score >> 20) == 5:
            nut_finishes += 0.5

    return wins / max(1, num_samples), nut_finishes / max(1, num_samples)


def compute_blocker_bonus(obs: Observation) -> float:
    """
    In Short Deck 5-Card PLO, a Flush requires EXACTLY 2 suited hole cards + 3 suited board cards.
    If the board has >= 3 cards of suit S, and we hold the Ace (rank 8) or King (rank 7) of suit S,
    we heavily block the opponent from holding the nut flush.
    """
    board = obs.board
    if len(board) < 3:
        return 0.0

    board_suit_counts = [0, 0, 0, 0]
    for c in board:
        board_suit_counts[c & 3] += 1

    bonus = 0.0
    for s in range(4):
        if board_suit_counts[s] >= 3:
            my_suited_ranks = [c >> 2 for c in obs.hole_cards if (c & 3) == s]
            # Even if we only have 1 card of suit s (no flush), holding Ace of suit s blocks nut flush!
            if 8 in my_suited_ranks:
                bonus += 0.08
            elif 7 in my_suited_ranks:
                bonus += 0.04
    return bonus


def choose_postflop_action(obs: Observation, prof: dict, num_samples: int) -> Action:
    # Determine how tight opponent's range is based on betting so far
    opp_tightness = 0.40
    if obs.to_call > 0 or obs.raises_this_street > 0:
        opp_tightness = 0.50 if not prof["is_maniac"] else 0.36

    equity, nut_rate = estimate_postflop_equity(obs, num_samples, opp_tightness)
    blocker_bonus = compute_blocker_bonus(obs)

    # Positional realization adjustment (acting last on button realizes more equity)
    pos_adj = 0.035 if obs.is_button else -0.020
    eff_equity = min(0.99, max(0.01, equity + pos_adj * (1.0 if obs.street != Street.RIVER else 0.0)))

    # Adjust equity based on opponent profile
    if prof["is_maniac"] and obs.to_call > 0:
        eff_equity = min(0.98, eff_equity + 0.065)
    elif prof["is_nit"] and obs.to_call > obs.big_blind * 2:
        eff_equity = max(0.01, eff_equity - 0.060)

    pot = float(obs.pot)
    to_call = float(obs.to_call)
    eff_stack = float(min(obs.my_stack, obs.opp_stack + obs.to_call))
    spr = eff_stack / max(1.0, pot)

    # CASE 1: We are facing a free check (to_call == 0)
    if obs.to_call == 0:
        # 2-PLY DEPTH-FIRST LOOKAHEAD: Should we Check-Trap or Bet Out?
        # Against maniacs when we have a monster (eff_equity > 0.78), checking induces a bluff!
        if prof["is_maniac"] and eff_equity > 0.78 and obs.street in (Street.FLOP, Street.TURN):
            if not obs.is_button and Action.CHECK_CALL in obs.legal_actions:
                # Check out-of-position to let the maniac bet into us, then raise!
                return Action.CHECK_CALL

        # Value betting with strong equity
        if eff_equity > 0.66 or (eff_equity > 0.56 and nut_rate > 0.35):
            if Action.RAISE_POT in obs.legal_actions and (eff_equity > 0.72 or prof["is_station"] or spr <= 1.8):
                return Action.RAISE_POT
            if Action.RAISE_HALF in obs.legal_actions:
                return Action.RAISE_HALF

        # Medium-strength value / protection bet
        if eff_equity > 0.54 and Action.RAISE_HALF in obs.legal_actions:
            return Action.RAISE_HALF

        # Exploitative Bluff / Semi-Bluff with Blockers or Redraws when opponent folds often
        fold_eq = prof["fold_to_bet"]
        if not prof["is_station"] and fold_eq > 0.44:
            bluff_score = blocker_bonus + 0.5 * nut_rate + (0.04 if obs.is_button else 0.0)
            if bluff_score >= 0.09 and Action.RAISE_HALF in obs.legal_actions:
                return Action.RAISE_HALF

        return Action.CHECK_CALL

    # CASE 2: We are facing a bet/raise (to_call > 0)
    pot_odds = to_call / (pot + to_call)

    # Compute EV of calling vs raising
    # If shallow SPR (<= 1.4) and we have strong equity + redraws, commit stack
    if eff_equity > 0.68 or (spr <= 1.4 and eff_equity > 0.54):
        if Action.RAISE_POT in obs.legal_actions and eff_equity > 0.70:
            return Action.RAISE_POT
        if Action.RAISE_HALF in obs.legal_actions and eff_equity > 0.64:
            return Action.RAISE_HALF
        return Action.CHECK_CALL

    # Semi-bluff raise against non-stations when holding massive nut redraw or blocker on small bet
    if not prof["is_station"] and prof["fold_to_bet"] > 0.50 and to_call <= pot * 0.55:
        if (nut_rate > 0.45 or blocker_bonus >= 0.08) and eff_equity > 0.42:
            if Action.RAISE_HALF in obs.legal_actions:
                return Action.RAISE_HALF

    # Call decision: compare effective equity against pot odds + implied odds buffer
    implied_buffer = 0.035 if (obs.is_button and obs.street != Street.RIVER and nut_rate > 0.25) else 0.0
    if eff_equity + implied_buffer >= pot_odds:
        return Action.CHECK_CALL

    if Action.FOLD in obs.legal_actions:
        return Action.FOLD
    return Action.CHECK_CALL
