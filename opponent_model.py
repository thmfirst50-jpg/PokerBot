from random import Random
from game import Action, Street, Observation, HandResult
from evaluator import omaha_score


def init_opponent_memory(memory: dict) -> None:
    memory["time_spent"] = 0.0
    memory["hands_seen"] = 0
    memory["last_processed_action_idx"] = 0
    memory["current_hand_idx"] = -1

    # Fast-adapting Bayesian Laplace-smoothed counters
    memory["pf_opp_acts"] = 5.0
    memory["pf_opp_raises"] = 1.8       # Prior ~36% preflop raise
    memory["pf_opp_folds"] = 1.2        # Prior ~24% preflop fold

    memory["post_opp_acts"] = 6.0
    memory["post_opp_raises"] = 1.8     # Prior ~30% postflop aggression
    memory["post_opp_faced_bet"] = 4.0
    memory["post_opp_folds"] = 1.4      # Prior ~35% fold-to-bet

    memory["showdowns"] = 1.0
    memory["opp_sd_cat_sum"] = 4.5      # Prior average category ~4.5 (Trips/Full House)

    # Rolling window of last 16 opponent actions and consecutive auto-folds
    memory["recent_opp_actions"] = []
    memory["consecutive_opp_folds"] = 0

    # Caches for stratified equity engine
    memory["rng"] = Random(202605)
    memory["board_key"] = None
    memory["eq_data"] = None


def update_live_action_stats(obs: Observation, memory: dict) -> None:
    if obs.hand_index != memory.get("current_hand_idx", -1):
        memory["current_hand_idx"] = obs.hand_index
        memory["last_processed_action_idx"] = 0

    history = obs.action_history
    start_idx = memory.get("last_processed_action_idx", 0)
    recent = memory.setdefault("recent_opp_actions", [])

    for i in range(start_idx, len(history)):
        ev = history[i]
        if not ev.is_me:
            is_raise = ev.action in (Action.RAISE_HALF, Action.RAISE_POT)
            is_fold = (ev.action == Action.FOLD)
            faced_bet = (
                i > 0
                and history[i - 1].is_me
                and history[i - 1].action in (Action.RAISE_HALF, Action.RAISE_POT)
            )

            # Record action in rolling window (tuple: is_raise, is_call_faced_bet, is_check, is_fold)
            is_call_faced_bet = (ev.action == Action.CHECK_CALL and faced_bet)
            is_check = (ev.action == Action.CHECK_CALL and not faced_bet)
            recent.append((is_raise, is_call_faced_bet, is_check, is_fold))
            if len(recent) > 16:
                recent.pop(0)

            # Active resistance resets consecutive folds
            if is_raise or is_call_faced_bet:
                memory["consecutive_opp_folds"] = 0

            if ev.street == Street.PREFLOP:
                memory["pf_opp_acts"] += 1.0
                if is_raise:
                    memory["pf_opp_raises"] += 1.0
                if is_fold:
                    memory["pf_opp_folds"] += 1.0
            else:
                memory["post_opp_acts"] += 1.0
                if is_raise:
                    memory["post_opp_raises"] += 1.0
                if faced_bet:
                    memory["post_opp_faced_bet"] += 1.0
                    if is_fold:
                        memory["post_opp_folds"] += 1.0

    memory["last_processed_action_idx"] = len(history)


def update_on_hand_end(result: HandResult, memory: dict) -> None:
    memory["hands_seen"] = memory.get("hands_seen", 0) + 1

    # Track opponent folds
    if not result.went_to_showdown and result.my_net > 0:
        memory["consecutive_opp_folds"] = memory.get("consecutive_opp_folds", 0) + 1
        recent = memory.setdefault("recent_opp_actions", [])
        recent.append((False, False, False, True))
        if len(recent) > 16:
            recent.pop(0)

        if len(result.board) == 0:
            memory["pf_opp_acts"] = memory.get("pf_opp_acts", 5.0) + 1.0
            memory["pf_opp_folds"] = memory.get("pf_opp_folds", 1.2) + 1.0
        else:
            memory["post_opp_acts"] = memory.get("post_opp_acts", 6.0) + 1.0
            memory["post_opp_faced_bet"] = memory.get("post_opp_faced_bet", 4.0) + 1.0
            memory["post_opp_folds"] = memory.get("post_opp_folds", 1.4) + 1.0
    else:
        memory["consecutive_opp_folds"] = 0

    if result.went_to_showdown and result.opp_hole is not None and len(result.board) == 5:
        opp_score = omaha_score(result.opp_hole, result.board)
        opp_cat = opp_score >> 20
        memory["showdowns"] = memory.get("showdowns", 1.0) + 1.0
        memory["opp_sd_cat_sum"] = memory.get("opp_sd_cat_sum", 4.5) + float(opp_cat)


def get_opponent_profile(memory: dict) -> dict:
    pf_acts = max(1.0, memory["pf_opp_acts"])
    pf_agg = memory["pf_opp_raises"] / pf_acts
    pf_fold_rate = memory["pf_opp_folds"] / pf_acts

    post_acts = max(1.0, memory["post_opp_acts"])
    post_agg = memory["post_opp_raises"] / post_acts
    post_faced = max(1.0, memory["post_opp_faced_bet"])
    fold_to_bet = memory["post_opp_folds"] / post_faced
    avg_sd_cat = memory["opp_sd_cat_sum"] / max(1.0, memory["showdowns"])

    # Fix 3: Dead-Clock / Time-out Detector
    # When opponent exhausts 8.0s time bank, judge auto-checks when free and auto-folds to bets
    recent = memory.get("recent_opp_actions", [])
    hand_idx = memory.get("current_hand_idx", 0)
    consec_folds = memory.get("consecutive_opp_folds", 0)
    is_clock_dead = False
    if hand_idx >= 350:
        if consec_folds >= 5:
            is_clock_dead = True
        elif len(recent) >= 10:
            active_resistance = any(r[0] or r[1] for r in recent)
            if not active_resistance:
                is_clock_dead = True

    # Fix 2: Distinguish blind spammers (AlwaysHalf/AlwaysPot) from thinking aggro bots (Riverglass)
    is_blind_spammer = (
        not is_clock_dead
        and (
            (pf_agg > 0.78 and post_agg > 0.75)
            or (pf_agg > 0.72 and fold_to_bet < 0.10 and memory.get("hands_seen", 0) >= 10)
        )
    )

    is_aggro_shark = (
        not is_clock_dead
        and not is_blind_spammer
        and (
            (post_agg > 0.22 and fold_to_bet > 0.28)
            or (pf_agg > 0.45 and fold_to_bet > 0.35)
        )
    )

    is_station = (
        not is_clock_dead
        and not is_blind_spammer
        and (fold_to_bet < 0.25)
        and (post_agg < 0.35)
    )
    is_nit = (fold_to_bet > 0.55) and (post_agg < 0.28)

    # Opponent range tightness for Monte Carlo sampling
    if is_blind_spammer or is_station:
        est_opp_tightness = 0.25  # 100% range, no filtering needed
    elif is_aggro_shark:
        est_opp_tightness = 0.50  # Tight-aggressive, respect their raises
    elif is_nit or avg_sd_cat > 5.2:
        est_opp_tightness = 0.54  # Very tight range
    elif avg_sd_cat < 3.6:
        est_opp_tightness = 0.35  # Loose range
    else:
        est_opp_tightness = 0.43  # Standard range

    return {
        "pf_agg": pf_agg,
        "post_agg": post_agg,
        "fold_to_bet": fold_to_bet,
        "avg_sd_cat": avg_sd_cat,
        "is_clock_dead": is_clock_dead,
        "is_blind_spammer": is_blind_spammer,
        "is_aggro_shark": is_aggro_shark,
        "is_station": is_station,
        "is_nit": is_nit,
        "est_opp_tightness": est_opp_tightness,
    }
