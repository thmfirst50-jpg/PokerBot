from game import Action, Street, Observation, HandResult
from evaluator import omaha_score


def init_opponent_memory(memory: dict) -> None:
    memory["time_spent"] = 0.0
    memory["hands_seen"] = 0
    memory["last_processed_action_idx"] = 0
    memory["current_hand_idx"] = -1

    # Bayesian Laplace-smoothed counters
    memory["pf_opp_acts"] = 12.0
    memory["pf_opp_raises"] = 4.0       # Prior 33% preflop raise
    memory["pf_opp_folds"] = 3.0        # Prior 25% preflop fold

    memory["post_opp_acts"] = 16.0
    memory["post_opp_raises"] = 5.0     # Prior 31% postflop aggression
    memory["post_opp_faced_bet"] = 12.0
    memory["post_opp_folds"] = 4.5      # Prior 37.5% fold-to-bet

    memory["showdowns"] = 2.0
    memory["opp_sd_cat_sum"] = 9.0      # Prior average category ~4.5 (Trips/Full House)


def update_live_action_stats(obs: Observation, memory: dict) -> None:
    if obs.hand_index != memory.get("current_hand_idx", -1):
        memory["current_hand_idx"] = obs.hand_index
        memory["last_processed_action_idx"] = 0

    history = obs.action_history
    start_idx = memory.get("last_processed_action_idx", 0)

    for i in range(start_idx, len(history)):
        ev = history[i]
        if not ev.is_me:
            is_raise = ev.action in (Action.RAISE_HALF, Action.RAISE_POT)
            is_fold = (ev.action == Action.FOLD)
            # Check if opponent was facing a bet/raise right before this action
            faced_bet = (
                i > 0
                and history[i - 1].is_me
                and history[i - 1].action in (Action.RAISE_HALF, Action.RAISE_POT)
            )

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
    if result.went_to_showdown and result.opp_hole is not None and len(result.board) == 5:
        opp_score = omaha_score(result.opp_hole, result.board)
        opp_cat = opp_score >> 20
        memory["showdowns"] = memory.get("showdowns", 2.0) + 1.0
        memory["opp_sd_cat_sum"] = memory.get("opp_sd_cat_sum", 9.0) + float(opp_cat)


def get_opponent_profile(memory: dict) -> dict:
    pf_agg = memory["pf_opp_raises"] / max(1.0, memory["pf_opp_acts"])
    post_agg = memory["post_opp_raises"] / max(1.0, memory["post_opp_acts"])
    fold_to_bet = memory["post_opp_folds"] / max(1.0, memory["post_opp_faced_bet"])
    avg_sd_cat = memory["opp_sd_cat_sum"] / max(1.0, memory["showdowns"])

    return {
        "pf_agg": pf_agg,               # High (>0.55) = preflop maniac
        "post_agg": post_agg,           # High (>0.50) = postflop bluffer/maniac
        "fold_to_bet": fold_to_bet,     # High (>0.52) = over-folder; Low (<0.25) = calling station
        "avg_sd_cat": avg_sd_cat,       # Low (<3.8) = shows up with weak hands
        "is_station": fold_to_bet < 0.25 and post_agg < 0.35,
        "is_maniac": post_agg > 0.52 or pf_agg > 0.60,
        "is_nit": fold_to_bet > 0.55 and post_agg < 0.28,
    }
