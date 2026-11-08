"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "BANG"
AGENT_NAME = "LarpGentV3"


# ===== YOUR CODE HERE =====

import time
from preflop_eval import compute_preflop_percentile
from opponent_model import (
    init_opponent_memory,
    update_live_action_stats,
    update_on_hand_end,
    get_opponent_profile,
)
from postflop_engine import choose_postflop_action


def _safe_fallback(obs: Observation) -> Action:
    if Action.CHECK_CALL in obs.legal_actions and obs.to_call == 0:
        return Action.CHECK_CALL
    if Action.FOLD in obs.legal_actions and obs.to_call > obs.big_blind * 2:
        return Action.FOLD
    return obs.legal_actions[0]


def _choose_preflop_action(obs: Observation, prof: dict) -> Action:
    # 0. Dead-Clock Auto-Print: When opponent is timed out / auto-folding, bet 100% of hands
    if prof.get("is_clock_dead", False):
        if Action.RAISE_HALF in obs.legal_actions:
            return Action.RAISE_HALF
        if Action.RAISE_POT in obs.legal_actions:
            return Action.RAISE_POT
        if Action.CHECK_CALL in obs.legal_actions:
            return Action.CHECK_CALL
        return obs.legal_actions[0]

    strength = compute_preflop_percentile(obs.hole_cards)

    # Positional adjustment: button acts last postflop
    if obs.is_button:
        strength += 0.035

    # 1. SPECIAL PREFLOP EXPLOIT VS BLIND SPAMMERS (Always-Half / Always-Pot bots)
    if prof.get("is_blind_spammer", False):
        # Punish their 100% raising range with top hands
        if strength >= 0.58:
            if Action.RAISE_POT in obs.legal_actions and strength >= 0.64:
                return Action.RAISE_POT
            if Action.RAISE_HALF in obs.legal_actions:
                return Action.RAISE_HALF
            return Action.CHECK_CALL
        # Call their raises with playable hands
        if strength >= 0.45 or obs.to_call <= obs.big_blind:
            return Action.CHECK_CALL
        if Action.FOLD in obs.legal_actions:
            return Action.FOLD
        return Action.CHECK_CALL

    # 2. SPECIAL PREFLOP EXPLOIT VS AGGRO SHARK (e.g. Riverglass / Hari)
    if prof.get("is_aggro_shark", False):
        # Open button with playable hands, fold bottom junk
        if obs.is_button and obs.to_call <= obs.big_blind:
            if strength >= 0.56 and Action.RAISE_POT in obs.legal_actions:
                return Action.RAISE_POT
            if strength >= 0.43 and Action.RAISE_HALF in obs.legal_actions:
                return Action.RAISE_HALF
            if strength < 0.36 and Action.FOLD in obs.legal_actions:
                return Action.FOLD
            if Action.CHECK_CALL in obs.legal_actions:
                return Action.CHECK_CALL
        # Big blind facing limp (call_cost == 0)
        elif not obs.is_button and obs.to_call == 0:
            if strength >= 0.56 and Action.RAISE_POT in obs.legal_actions:
                return Action.RAISE_POT
            if strength >= 0.48 and Action.RAISE_HALF in obs.legal_actions:
                return Action.RAISE_HALF
            return Action.CHECK_CALL
        # Facing a raise: 3-bet for value with premium hands, fold dominated trash
        elif obs.to_call > obs.big_blind:
            pot_odds = obs.to_call / float(obs.pot + obs.to_call)
            # Facing an open-raise (raises_this_street == 1)
            if obs.raises_this_street == 1:
                if strength >= 0.64 and Action.RAISE_POT in obs.legal_actions:
                    return Action.RAISE_POT
                if strength >= 0.58 and Action.RAISE_HALF in obs.legal_actions:
                    return Action.RAISE_HALF
                # Defend with playable hands, fold junk OOP
                req = max(0.43, pot_odds + 0.10 - (0.03 if obs.is_button else 0.0))
                if strength >= req:
                    return Action.CHECK_CALL
                if Action.FOLD in obs.legal_actions:
                    return Action.FOLD
                return Action.CHECK_CALL
            # Facing a 3-bet or 4-bet (raises_this_street >= 2)
            else:
                if obs.to_call >= obs.big_blind * 8:
                    # 4-bet shove defense: only continue with monster hands (AAxx / double pair)
                    if strength >= 0.82 and Action.RAISE_POT in obs.legal_actions:
                        return Action.RAISE_POT
                    if strength >= 0.60:
                        return Action.CHECK_CALL
                    if Action.FOLD in obs.legal_actions:
                        return Action.FOLD
                    return Action.CHECK_CALL
                else:
                    if strength >= 0.76 and Action.RAISE_POT in obs.legal_actions:
                        return Action.RAISE_POT
                    req = max(0.48, pot_odds + 0.12 - (0.02 if obs.is_button else 0.0))
                    if strength >= req:
                        return Action.CHECK_CALL
                    if Action.FOLD in obs.legal_actions:
                        return Action.FOLD
                    return Action.CHECK_CALL

    # 3. Standard opponent adjustments
    if prof.get("is_nit", False) and obs.raises_this_street >= 1:
        strength -= 0.040

    # Unopened pot or limped pot (to_call <= 2)
    if obs.to_call <= obs.big_blind:
        if strength >= 0.57 and Action.RAISE_POT in obs.legal_actions:
            return Action.RAISE_POT
        if strength >= 0.48 and Action.RAISE_HALF in obs.legal_actions:
            return Action.RAISE_HALF
        if obs.to_call == 0:
            return Action.CHECK_CALL
        if obs.to_call == 1 and strength >= 0.35:
            return Action.CHECK_CALL
        if Action.FOLD in obs.legal_actions:
            return Action.FOLD
        return Action.CHECK_CALL

    # Facing a preflop raise (to_call > 2)
    pot_odds = obs.to_call / float(obs.pot + obs.to_call)

    if obs.raises_this_street == 1:
        if strength >= 0.64 and Action.RAISE_POT in obs.legal_actions:
            return Action.RAISE_POT
        if strength >= 0.58 and Action.RAISE_HALF in obs.legal_actions:
            return Action.RAISE_HALF
        required_strength = max(0.43, pot_odds + 0.11 - (0.03 if obs.is_button else 0.0))
    else:
        if strength >= 0.72 and Action.RAISE_POT in obs.legal_actions:
            return Action.RAISE_POT
        required_strength = max(0.49, pot_odds + 0.14 - (0.02 if obs.is_button else 0.0))

    if strength >= required_strength:
        return Action.CHECK_CALL

    if Action.FOLD in obs.legal_actions:
        return Action.FOLD
    return Action.CHECK_CALL


def new_half(config, memory: dict) -> None:
    try:
        init_opponent_memory(memory)
    except Exception:
        pass


def choose_action(obs: Observation, memory: dict) -> Action:
    t0 = time.perf_counter()
    try:
        if "time_spent" not in memory:
            init_opponent_memory(memory)

        update_live_action_stats(obs, memory)
        prof = get_opponent_profile(memory)

        if obs.street == Street.PREFLOP:
            action = _choose_preflop_action(obs, prof)
        else:
            # Dynamic Time-Bank Allocation (8.0s limit per 1,000 hands; target 3.8s-4.8s)
            spent = memory.get("time_spent", 0.0)
            hand_idx = obs.hand_index
            target_spent = 4.5 * (hand_idx / 1000.0)
            pot = obs.pot
            is_big_pot = (pot >= 12)

            # Base sample budget by street (optimized for speed + accuracy)
            if obs.street == Street.RIVER:
                base_samples = 40 if is_big_pot else 26
            elif obs.street == Street.TURN:
                base_samples = 27
            else:  # Flop
                base_samples = 20 if is_big_pot else 14

            # Dynamic pacing adjustments to strictly prevent timing out
            if spent > 5.5:
                samples = 10
            elif spent > 4.5:
                samples = 16
            elif spent > target_spent + 0.4:
                samples = 18 if is_big_pot else 12
            else:
                samples = base_samples

            action = choose_postflop_action(obs, prof, samples, memory)

        if action not in obs.legal_actions:
            action = _safe_fallback(obs)
    except Exception:
        action = _safe_fallback(obs)
    finally:
        # Accounting for execution time plus Windows localhost HTTP overhead (~2.2ms - 1ms grace = +1.3ms)
        compute_time = time.perf_counter() - t0
        charged_est = compute_time + 0.0013
        memory["time_spent"] = memory.get("time_spent", 0.0) + charged_est

    return action


def on_hand_end(result: HandResult, memory: dict) -> None:
    try:
        if "time_spent" not in memory:
            init_opponent_memory(memory)
        update_on_hand_end(result, memory)
    except Exception:
        pass

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
