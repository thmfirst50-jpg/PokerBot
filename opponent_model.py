from random import Random
from game import Action, Street, Observation, HandResult
from evaluator import omaha_score


def init_opponent_memory(memory: dict) -> None:
    memory["time_spent"] = 0.0
    memory["hands_seen"] = 0
    memory["last_processed_action_idx"] = 0
    memory["current_hand_idx"] = -1

    # Continuous Street-by-Street Counters (0: Preflop, 1: Flop, 2: Turn, 3: River)
    # Smooth Laplace priors
    memory["street_acts"] = [4.0, 4.0, 3.0, 2.5]
    memory["street_raises"] = [1.4, 1.2, 0.9, 0.75]
    memory["street_pot_raises"] = [0.7, 0.5, 0.4, 0.35]
    memory["street_faced_bet"] = [2.0, 2.5, 2.0, 1.8]
    memory["street_folds"] = [0.6, 0.9, 0.7, 0.65]

    # Showdown Bet-Sizing Credibility
    memory["opp_agg_sd_count"] = 3.0
    memory["opp_agg_sd_sum"] = 3.0 * 0.68  # Prior ~0.68 percentile
    memory["opp_raised_postflop_this_hand"] = False

    # Passive Clock-Out Detector
    memory["consecutive_passive_actions"] = 0

    # Caches for stratified equity engine
    memory["rng"] = Random(202605)
    memory["board_key"] = None
    memory["eq_data"] = None


def update_live_action_stats(obs: Observation, memory: dict) -> None:
    if obs.hand_index != memory.get("current_hand_idx", -1):
        memory["current_hand_idx"] = obs.hand_index
        memory["last_processed_action_idx"] = 0
        memory["opp_raised_postflop_this_hand"] = False

    history = obs.action_history
    start_idx = memory.get("last_processed_action_idx", 0)

    for i in range(start_idx, len(history)):
        ev = history[i]
        if not ev.is_me:
            s = ev.street.value
            is_raise = ev.action in (Action.RAISE_HALF, Action.RAISE_POT)
            is_pot_raise = (ev.action == Action.RAISE_POT)
            faced_bet = (
                i > 0
                and history[i - 1].is_me
                and history[i - 1].action in (Action.RAISE_HALF, Action.RAISE_POT)
            )

            memory["street_acts"][s] += 1.0
            if is_raise:
                memory["street_raises"][s] += 1.0
                if s != 0:
                    memory["opp_raised_postflop_this_hand"] = True
            if is_pot_raise:
                memory["street_pot_raises"][s] += 1.0
            if faced_bet:
                memory["street_faced_bet"][s] += 1.0

            # Passive clock-out detector:
            # Active resistance (raise or call faced bet with amount > 0) resets passive counter
            if is_raise or ev.amount > 0:
                memory["consecutive_passive_actions"] = 0
            else:
                memory["consecutive_passive_actions"] += 1

    memory["last_processed_action_idx"] = len(history)


def update_on_hand_end(result: HandResult, memory: dict) -> None:
    memory["hands_seen"] = memory.get("hands_seen", 0) + 1

    # Opponent folded (ends hand immediately without showdown)
    if not result.went_to_showdown and result.my_net > 0:
        memory["consecutive_passive_actions"] = memory.get("consecutive_passive_actions", 0) + 1
        b_len = len(result.board)
        if b_len == 0:
            s = 0
        elif b_len == 3:
            s = 1
        elif b_len == 4:
            s = 2
        else:
            s = 3
        memory["street_acts"][s] += 1.0
        memory["street_faced_bet"][s] += 1.0
        memory["street_folds"][s] += 1.0
    elif result.went_to_showdown and result.pot > 4:
        # Showdown reached in a raised pot: opponent called or raised
        memory["consecutive_passive_actions"] = 0
    elif result.my_net < 0:
        # Opponent won chips from us: opponent is active
        memory["consecutive_passive_actions"] = 0

    # Showdown bet-sizing credibility
    if result.went_to_showdown and result.opp_hole is not None and len(result.board) == 5:
        if memory.get("opp_raised_postflop_this_hand", False):
            board = list(result.board)
            opp_hole = list(result.opp_hole)
            opp_score = omaha_score(opp_hole, board)
            known = set(board) | set(opp_hole)
            avail = [c for c in range(36) if c not in known]
            rng = memory.get("rng") or Random(202605)
            n_samples = 25
            wins = 0.0
            for _ in range(n_samples):
                smpl = rng.sample(avail, 5)
                sc = omaha_score(smpl, board)
                if opp_score > sc:
                    wins += 1.0
                elif opp_score == sc:
                    wins += 0.5
            p_sd = wins / n_samples
            memory["opp_agg_sd_count"] += 1.0
            memory["opp_agg_sd_sum"] += p_sd

    memory["opp_raised_postflop_this_hand"] = False


def get_opponent_profile(memory: dict) -> dict:
    acts = memory["street_acts"]
    raises = memory["street_raises"]
    pot_raises = memory["street_pot_raises"]
    faced = memory["street_faced_bet"]
    folds = memory["street_folds"]

    agg_freq = [raises[s] / max(1.0, acts[s]) for s in range(4)]
    pot_bet_freq = [pot_raises[s] / max(1.0, acts[s]) for s in range(4)]
    fold_freq = [folds[s] / max(1.0, faced[s]) for s in range(4)]
    opp_agg_sd_percentile = memory["opp_agg_sd_sum"] / max(1.0, memory["opp_agg_sd_count"])

    hand_idx = memory.get("current_hand_idx", 0)
    passive_actions = memory.get("consecutive_passive_actions", 0)
    is_auto_folding = (hand_idx >= 500) and (passive_actions >= 18)

    pf_agg = agg_freq[0]
    post_agg = sum(raises[1:]) / max(1.0, sum(acts[1:]))
    fold_to_bet = sum(folds[1:]) / max(1.0, sum(faced[1:]))

    # Universal detection for blind spammers (100% raisers like AlwaysHalf/AlwaysPot)
    is_blind_spammer = (
        not is_auto_folding
        and (
            (agg_freq[0] > 0.78 and agg_freq[1] > 0.75)
            or (agg_freq[0] > 0.72 and fold_freq[1] < 0.10 and acts[0] >= 10)
        )
    )

    is_station = (
        not is_auto_folding
        and not is_blind_spammer
        and (fold_to_bet < 0.20 and post_agg < 0.25)
    )

    # Range tightness estimate for sampling
    if is_blind_spammer or is_station:
        est_opp_tightness = 0.25
    elif agg_freq[0] > 0.45 and fold_to_bet > 0.30:
        est_opp_tightness = 0.50
    elif fold_to_bet > 0.50 and post_agg < 0.28:
        est_opp_tightness = 0.55
    else:
        est_opp_tightness = 0.42

    return {
        "agg_freq": agg_freq,
        "pot_bet_freq": pot_bet_freq,
        "fold_freq": fold_freq,
        "opp_agg_sd_percentile": opp_agg_sd_percentile,
        "is_auto_folding": is_auto_folding,
        "is_clock_dead": is_auto_folding,
        "is_blind_spammer": is_blind_spammer,
        "is_station": is_station,
        "pf_agg": pf_agg,
        "post_agg": post_agg,
        "fold_to_bet": fold_to_bet,
        "est_opp_tightness": est_opp_tightness,
    }
