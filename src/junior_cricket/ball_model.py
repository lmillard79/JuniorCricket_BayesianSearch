"""Joint batter-by-bowler model of ball outcomes.

The per-player model estimates a batter's scoring and a bowler's economy
separately, so each absorbs the quality of the opposition the other side
faced. This model works ball by ball instead. For every delivery it
records who batted and who bowled, and each outcome probability depends
on both, additively on the logit scale:

* ``d`` dismissal (all balls);
* ``b`` boundary (balls that are not dismissals);
* ``s`` scoring ball, meaning at least one run (balls that are neither).

Each player has a batting effect and a bowling effect for each of these,
drawn from a common spread (non-centred), so a batter's boundary power
is measured net of the bowlers faced and a bowler's economy net of the
batters faced. Boundary and scoring outcomes also share a per-game
effect (ground and conditions). Two-run and three-run balls are modelled
by fixed shares from the data.

``JointOutcomes`` turns a fitted posterior into the per-ball rule the
innings engines use, in place of their hand-built combination rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:       # PyMC is imported where it is used, so simulation workers stay light
    import pymc as pm

COMPONENTS = ("d", "b", "s")
LABELS = {
    "d": "dismissal (per ball)",
    "b": "boundary (per ball that is not a dismissal)",
    "s": "scoring shot (per non-boundary, non-dismissal ball)",
}
# Prior centres on the probability scale, from the 2025/26 U10 games.
CENTRES = {"d": 0.055, "b": 0.10, "s": 0.50}
BOUNDARY_RUN_PROBS = ((4, 0.875), (5, 0.035), (6, 0.090))
P_TWO_GIVEN_SCORING = 0.124
P_THREE_GIVEN_TWO_PLUS = 0.072


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _invlogit(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class BallData:
    """Every delivery as integer-coded arrays.

    Attributes:
        players: Player labels; index order used by ``bat`` and ``bowl``.
        games: Game labels; index order used by ``game``.
        bat: Batter index per ball.
        bowl: Bowler index per ball.
        game: Game index per ball.
        runs: Runs recorded off the ball.
        dismissed: 1 if the ball was a dismissal.
        boundary: 1 if the ball was a four, five or six.
        ours_fielding: 1 if the bowler (and so the whole fielding side)
            was our team, 0 for the opposition. Drives the fielding
            effect, which only ever applies to our own side.
    """

    players: List[str]
    games: List[str]
    bat: np.ndarray
    bowl: np.ndarray
    game: np.ndarray
    runs: np.ndarray
    dismissed: np.ndarray
    boundary: np.ndarray
    ours_fielding: np.ndarray

    def component(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Mask of balls that inform a component, and their outcomes.

        Args:
            name: ``d``, ``b`` or ``s``.

        Returns:
            (boolean mask over all balls, outcome 0/1 for masked balls).
        """
        if name == "d":
            mask = np.ones(len(self.bat), dtype=bool)
            return mask, self.dismissed[mask]
        not_out = self.dismissed == 0
        if name == "b":
            return not_out, self.boundary[not_out]
        mask = not_out & (self.boundary == 0)
        return mask, (self.runs[mask] >= 1).astype(int)

    def take(self, mask: np.ndarray) -> "BallData":
        """The same coding restricted to some balls."""
        return BallData(
            self.players, self.games, self.bat[mask], self.bowl[mask],
            self.game[mask], self.runs[mask], self.dismissed[mask],
            self.boundary[mask], self.ours_fielding[mask],
        )


OUR_ALIAS_PREFIX = "P"   # our own players; the opposition's aliases start with O


def load_balls(
    path: Path, our_names: Optional[Sequence[str]] = None,
) -> Tuple[BallData, pd.DataFrame]:
    """Read ``playhq_balls.csv`` into integer-coded arrays.

    Args:
        path: CSV written by ``fetch_scorecards.py``.
        our_names: Exactly who "ours" is, for the fielding effect
            (``ours_fielding``). Defaults to the ``P0x``/``O0x`` alias
            convention (every batter/bowler string is required to start
            with one of those two letters); pass the real roster instead
            when the CSV uses real names rather than aliases (a "named"
            data-dir built for a coach to sanity-check against what they
            watched), since names carry no such prefix to lean on.

    Returns:
        (coded data, the raw frame with its ``date`` column parsed).

    Raises:
        ValueError: If ``our_names`` is None and a bowler string is
            neither ours nor the opposition's by the alias convention,
            so ``ours_fielding`` would otherwise be silently wrong.
    """
    frame = pd.read_csv(path, parse_dates=["date"], dtype={"fielder": str})
    frame["fielder"] = frame["fielder"].fillna("")
    players = sorted(set(frame["batter"]) | set(frame["bowler"]))
    games = list(dict.fromkeys(frame["game_id"]))
    index = {p: i for i, p in enumerate(players)}
    gindex = {g: i for i, g in enumerate(games)}
    if our_names is not None:
        ours_mask = frame["bowler"].isin(set(our_names))
    else:
        bowler_prefix = frame["bowler"].str[0]
        stray = sorted(set(frame.loc[~bowler_prefix.isin(["P", "O"]), "bowler"]))
        if stray:
            raise ValueError(f"Bowler aliases outside the P/O convention: {stray}")
        ours_mask = bowler_prefix == OUR_ALIAS_PREFIX
    data = BallData(
        players=players,
        games=games,
        bat=frame["batter"].map(index).to_numpy(),
        bowl=frame["bowler"].map(index).to_numpy(),
        game=frame["game_id"].map(gindex).to_numpy(),
        runs=frame["runs"].to_numpy(),
        dismissed=frame["dismissed"].to_numpy(),
        boundary=frame["boundary"].to_numpy(),
        ours_fielding=ours_mask.to_numpy().astype(np.int8),
    )
    return data, frame


def build_ball_model(
    data: BallData,
    batter_effects: bool = True,
    bowler_effects: bool = True,
    game_effects: bool = True,
    fielding_effects: bool = True,
) -> pm.Model:
    """Build the joint model.

    Args:
        data: Coded deliveries.
        batter_effects: Include a batting effect per player.
        bowler_effects: Include a bowling effect per player.
        game_effects: Include a per-game effect on boundary and scoring,
            shared by both sides batting that day (ground and conditions).
        fielding_effects: Include a per-game effect on dismissal, specific
            to our own side's fielding that day (catches, run outs, general
            sharpness), on top of the bowler's own effect. Zero for the
            opposition's bowling, since only our side's fielding quality is
            of interest and PlayHQ's data cannot attribute it to individual
            fielders reliably enough to go further than a team-day level
            (see ``fielding_credit_table`` for the individual counts that
            are available, kept descriptive rather than fitted). Its group
            mean (``mu_f_d``) is the systematic edge across all games; its
            per-game draws (``field_d``) are the day-to-day wobble around
            that, in the same non-centred style as ``game_effects``.

    Returns:
        A ``pm.Model``. Switching the effect flags off gives the simpler
        variants the backtest compares against.
    """
    import pymc as pm

    coords = {"player": data.players, "game": data.games}
    with pm.Model(coords=coords) as model:
        for name in COMPONENTS:
            mask, y = data.component(name)
            eta = pm.Normal(f"a_{name}", _logit(CENTRES[name]), 0.7)
            if batter_effects:
                scale = pm.HalfNormal(f"sb_{name}", 0.6)
                z = pm.Normal(f"zb_{name}", 0.0, 1.0, dims="player")
                bat = pm.Deterministic(f"bat_{name}", scale * z, dims="player")
                eta = eta + bat[data.bat[mask]]
            if bowler_effects:
                scale = pm.HalfNormal(f"sw_{name}", 0.4)
                z = pm.Normal(f"zw_{name}", 0.0, 1.0, dims="player")
                bowl = pm.Deterministic(f"bowl_{name}", scale * z, dims="player")
                eta = eta + bowl[data.bowl[mask]]
            if game_effects and name in ("b", "s"):
                scale = pm.HalfNormal(f"sg_{name}", 0.3)
                z = pm.Normal(f"zg_{name}", 0.0, 1.0, dims="game")
                game = pm.Deterministic(f"game_{name}", scale * z, dims="game")
                eta = eta + game[data.game[mask]]
            if fielding_effects and name == "d":
                mu_f = pm.Normal("mu_f_d", 0.0, 0.3)
                scale_f = pm.HalfNormal("sf_d", 0.3)
                zf = pm.Normal("zf_d", 0.0, 1.0, dims="game")
                field = pm.Deterministic("field_d", mu_f + scale_f * zf, dims="game")
                eta = eta + field[data.game[mask]] * data.ours_fielding[mask]
            pm.Bernoulli(f"y_{name}", logit_p=eta, observed=y)
    return model


def fit_ball_model(
    data: BallData, draws: int = 1000, tune: int = 1000, chains: int = 4,
    seed: int = 0, **flags: bool,
):
    """Fit the joint model with the JAX sampler (no C compiler needed)."""
    import pymc as pm

    model = build_ball_model(data, **flags)
    with model:
        return pm.sample(
            draws=draws, tune=tune, chains=chains, target_accept=0.9,
            random_seed=seed, progressbar=False, nuts_sampler="numpyro",
        )


def _flat(idata, var: str) -> np.ndarray:
    """Posterior draws of ``var`` stacked as (draws, ...)."""
    array = np.asarray(idata.posterior[var])
    return array.reshape((-1,) + array.shape[2:])


def _posterior_arrays(idata) -> Dict[str, np.ndarray]:
    """Every variable ``JointOutcomes`` reads, flattened once."""
    names = [f"{p}_{c}" for p in ("a", "sb", "sw", "bat", "bowl") for c in COMPONENTS]
    names += ["sg_b", "sg_s", "game_b", "game_s", "mu_f_d", "sf_d", "field_d"]
    return {n: _flat(idata, n) for n in names if n in idata.posterior}


def _posterior_labels(idata) -> Tuple[List[str], List[str]]:
    """(player labels, game labels) in the posterior's own order."""
    players = [str(p) for p in idata.posterior["bat_d"].coords["player"].values]
    game_source = "game_b" if "game_b" in idata.posterior else "field_d"
    games = ([str(g) for g in idata.posterior[game_source].coords["game"].values]
             if game_source in idata.posterior else [])
    return players, games


def held_out_lpd(idata, test: BallData, name: str, variant: str) -> float:
    """Log predictive density of held-out balls for one component.

    Players and games unseen in training use their prior spread, so the
    prediction carries the right uncertainty.

    Args:
        idata: Posterior from ``fit_ball_model`` on earlier games.
        test: Held-out balls, coded on the same player list.
        name: Component ``d``, ``b`` or ``s``.
        variant: ``null`` (intercept only), ``batter`` or ``both``.

    Returns:
        Sum of log predictive density over the held-out balls.
    """
    mask, y = test.component(name)
    if not mask.any():
        return 0.0
    eta = _flat(idata, f"a_{name}")[:, None] + np.zeros((1, int(mask.sum())))
    if variant in ("batter", "both"):
        eta = eta + _flat(idata, f"bat_{name}")[:, test.bat[mask]]
    if variant == "both":
        eta = eta + _flat(idata, f"bowl_{name}")[:, test.bowl[mask]]
    p = np.clip(_invlogit(eta), 1e-9, 1 - 1e-9)
    per_ball = np.where(y == 1, p, 1.0 - p)                   # (draws, balls)
    return float(np.log(per_ball.mean(axis=0)).sum())


class JointOutcomes:
    """Per-ball outcome rule from a fitted joint model.

    The innings engines call ``dismissal_probability`` and ``runs`` for
    every ball, so batter and bowler enter together and a bowler's
    economy is no longer applied on top of a batter's scoring rate that
    already reflects the bowlers they met.

    Args:
        intercepts: ``a_d``, ``a_b``, ``a_s`` intercepts (logit).
        bat: Player -> {component: batting effect}.
        bowl: Player -> {component: bowling effect}.
        bat_sd: Component -> spread of batting effects (for new players).
        bowl_sd: Component -> spread of bowling effects.
        game_sd: Component (``b``, ``s``) -> spread of game effects.
        fielding_mu: Our team's average fielding edge on dismissal
            (logit scale); 0.0 when the posterior was fit without it.
        fielding_sd: Game-to-day spread of that edge around
            ``fielding_mu``.
        days: Game ID -> fitted per-day effects: ``b``/``s`` conditions
            and ``field_d`` (our fielding edge that day), for replaying
            a real day (see ``use_day``).
        draw: Posterior draw these effects came from; ``None`` for means.
    """

    def __init__(
        self,
        intercepts: Dict[str, float],
        bat: Dict[str, Dict[str, float]],
        bowl: Dict[str, Dict[str, float]],
        bat_sd: Dict[str, float],
        bowl_sd: Dict[str, float],
        game_sd: Dict[str, float],
        fielding_mu: float = 0.0,
        fielding_sd: float = 0.0,
        days: Optional[Dict[str, Dict[str, float]]] = None,
        draw: Optional[int] = None,
    ) -> None:
        self.a = intercepts
        self.bat = {k: dict(v) for k, v in bat.items()}
        self.bowl = {k: dict(v) for k, v in bowl.items()}
        self.bat_sd, self.bowl_sd, self.game_sd = bat_sd, bowl_sd, game_sd
        self.fielding_mu, self.fielding_sd = fielding_mu, fielding_sd
        self.game = {"b": 0.0, "s": 0.0}
        self.field_today = 0.0
        # Whether the dismissal effect currently includes our fielding
        # edge; toggle with set_fielding_ours before simulating an innings.
        self.fielding_ours = False
        self.days = days or {}
        self.draw = draw

    @classmethod
    def from_posterior(cls, idata, draw: Optional[int] = None) -> "JointOutcomes":
        """Build from posterior means, or from a single posterior draw.

        Args:
            idata: Posterior from ``fit_ball_model``.
            draw: Index into the stacked chain-and-draw samples. ``None``
                averages over all draws (the plug-in rule). An index gives
                one plausible set of effects; simulating with a pool of
                these (see ``pool``) carries the uncertainty about each
                player's skill into the results.
        """
        return cls._build(_posterior_arrays(idata), _posterior_labels(idata), draw)

    @classmethod
    def pool(cls, idata, size: int, rng: np.random.Generator) -> List["JointOutcomes"]:
        """Rules built from ``size`` distinct random posterior draws."""
        arrays, labels = _posterior_arrays(idata), _posterior_labels(idata)
        total = len(arrays["a_d"])
        picks = rng.choice(total, size=min(size, total), replace=False)
        return [cls._build(arrays, labels, int(i)) for i in picks]

    @classmethod
    def _build(cls, arrays, labels, draw: Optional[int]) -> "JointOutcomes":
        players, games = labels
        pick = (lambda a: a.mean(axis=0)) if draw is None else (lambda a: a[draw])   # noqa: E731
        scalar = lambda var: float(pick(arrays[var]))                                # noqa: E731
        bat = {p: {} for p in players}
        bowl = {p: {} for p in players}
        for c in COMPONENTS:
            for prefix, table in (("bat", bat), ("bowl", bowl)):
                for p, v in zip(players, pick(arrays[f"{prefix}_{c}"])):
                    table[p][c] = float(v)
        days: Dict[str, Dict[str, float]] = {}
        if "game_b" in arrays:
            gb, gs = pick(arrays["game_b"]), pick(arrays["game_s"])
            for g, b, s in zip(games, gb, gs):
                days.setdefault(g, {}).update({"b": float(b), "s": float(s)})
        if "field_d" in arrays:
            for g, f in zip(games, pick(arrays["field_d"])):
                days.setdefault(g, {})["field_d"] = float(f)
        return cls(
            intercepts={c: scalar(f"a_{c}") for c in COMPONENTS},
            bat=bat, bowl=bowl,
            bat_sd={c: scalar(f"sb_{c}") for c in COMPONENTS},
            bowl_sd={c: scalar(f"sw_{c}") for c in COMPONENTS},
            game_sd={c: scalar(f"sg_{c}") for c in ("b", "s")},
            fielding_mu=scalar("mu_f_d") if "mu_f_d" in arrays else 0.0,
            fielding_sd=scalar("sf_d") if "sf_d" in arrays else 0.0,
            days=days, draw=draw,
        )

    def use_day(self, game_id: str) -> None:
        """Fix the ground and conditions effect, and our fielding edge, at a game's fitted value."""
        day = self.days[game_id]
        self.game = {k: v for k, v in day.items() if k != "field_d"}
        self.field_today = day.get("field_d", self.fielding_mu)

    def register(self, name: str, rng: np.random.Generator) -> None:
        """Draw effects for a player not in the fitted data (an unknown opponent)."""
        self.bat[name] = {c: float(rng.normal(0, self.bat_sd[c])) for c in COMPONENTS}
        self.bowl[name] = {c: float(rng.normal(0, self.bowl_sd[c])) for c in COMPONENTS}

    def begin_game(self, rng: np.random.Generator) -> None:
        """Draw this game's ground and conditions effect, and our fielding edge."""
        self.game = {c: float(rng.normal(0, self.game_sd[c])) for c in ("b", "s")}
        self.field_today = float(rng.normal(self.fielding_mu, self.fielding_sd))

    def set_fielding_ours(self, ours: bool) -> None:
        """Toggle whether the dismissal effect includes our fielding edge.

        Call before simulating an innings: True while our team bowls,
        False while the opposition bowls. Defaults to False, so a caller
        that never calls this (the batting-order strategy comparison
        never simulates our team bowling) is unaffected.
        """
        self.fielding_ours = ours

    def _eta(self, component: str, striker, bowler) -> float:
        base = (self.a[component] + self.bat[striker.name][component]
                + self.bowl[bowler.name][component] + self.game.get(component, 0.0))
        if component == "d" and self.fielding_ours:
            base += self.field_today
        return base

    def dismissal_probability(self, striker, bowler) -> float:
        """Probability the ball is a dismissal."""
        return float(_invlogit(self._eta("d", striker, bowler)))

    def runs(self, striker, bowler, rng: np.random.Generator) -> Tuple[int, bool]:
        """Draw runs off a ball that is not a dismissal, and whether it is a boundary."""
        if rng.random() < _invlogit(self._eta("b", striker, bowler)):
            u, total = rng.random(), 0.0
            for runs, share in BOUNDARY_RUN_PROBS:
                total += share
                if u < total:
                    return runs, True
            return BOUNDARY_RUN_PROBS[-1][0], True
        if rng.random() < _invlogit(self._eta("s", striker, bowler)):
            if rng.random() < P_TWO_GIVEN_SCORING:
                return (3 if rng.random() < P_THREE_GIVEN_TWO_PLUS else 2), False
            return 1, False
        return 0, False


def _expected_boundary_runs() -> float:
    return sum(runs * share for runs, share in BOUNDARY_RUN_PROBS)


def _expected_scoring_runs() -> float:
    """Mean runs on a non-boundary scoring ball (1, 2 or 3)."""
    two = P_TWO_GIVEN_SCORING
    three = two * P_THREE_GIVEN_TWO_PLUS
    return (1 - two) * 1 + (two - three) * 2 + three * 3


def player_profiles(
    idata, players: Sequence[str], per_balls: int = 13, per_overs: int = 1,
) -> pd.DataFrame:
    """Plain-language batting and bowling profile for each player.

    Batting: expected outcomes over ``per_balls`` balls against an average
    bowler on an average ground. Bowling: expected wickets and runs over
    ``per_overs`` overs against an average batter. ``net`` counts each
    dismissal as 4 runs, the U10 penalty (Rule 16.10(vi)); it is
    indicative for U11, where a dismissal ends the innings instead.

    Args:
        idata: Posterior from ``fit_ball_model``.
        players: Players to describe.
        per_balls: Balls faced to scale batting figures to.
        per_overs: Overs bowled to scale bowling figures to.

    Returns:
        One row per player with posterior means and 90% intervals for the
        headline figures.
    """
    names = [str(p) for p in idata.posterior["bat_d"].coords["player"].values]
    a = {c: _flat(idata, f"a_{c}") for c in COMPONENTS}
    e_b, e_s = _expected_boundary_runs(), _expected_scoring_runs()

    def rates(bat, bowl):
        pd_ = _invlogit(a["d"] + bat["d"] + bowl["d"])
        pb = _invlogit(a["b"] + bat["b"] + bowl["b"])
        ps = _invlogit(a["s"] + bat["s"] + bowl["s"])
        runs = (1 - pd_) * (pb * e_b + (1 - pb) * ps * e_s)
        return pd_, (1 - pd_) * pb, runs

    zero = {c: 0.0 for c in COMPONENTS}
    avg_d, _, avg_runs = rates(zero, zero)
    avg_net = (avg_runs - 4 * avg_d).mean()
    rows = []
    for p in players:
        i = names.index(p)
        bat = {c: _flat(idata, f"bat_{c}")[:, i] for c in COMPONENTS}
        bowl = {c: _flat(idata, f"bowl_{c}")[:, i] for c in COMPONENTS}
        d, boundaries, runs = rates(bat, zero)
        bd, _, bruns = rates(zero, bowl)
        net = per_balls * (runs - 4 * d)
        wickets, conceded = 6 * per_overs * bd, 6 * per_overs * bruns
        edge = 6 * per_overs * ((4 * bd - bruns) - (4 * avg_d - avg_runs)).mean()

        def ci(x):
            return float(np.percentile(x, 5)), float(np.percentile(x, 95))

        rows.append({
            "player": p,
            "boundaries": per_balls * boundaries.mean(),
            "runs": per_balls * runs.mean(),
            "dismissals": per_balls * d.mean(),
            "net": net.mean(), "net_lo": ci(net)[0], "net_hi": ci(net)[1],
            "net_vs_average": net.mean() - per_balls * avg_net,
            "wickets": wickets.mean(), "wickets_lo": ci(wickets)[0], "wickets_hi": ci(wickets)[1],
            "runs_conceded": conceded.mean(),
            "bowling_edge": float(edge),
        })
    return pd.DataFrame(rows)


def fielding_credit_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Catches and run outs credited to each fielder, from the ball rows.

    Descriptive, not fitted: a handful of events per player across a
    season is a count worth showing a coach, not a modelled skill (unlike
    the fielding effect in the joint model, which only goes as far as a
    team-day level; see ``build_ball_model``). It also understates each
    player's fielding involvement, since only 67 of 172 dismissals in the
    2025/26 data name a fielder at all (a "bowled" dismissal credits no
    fielder even though eight team mates were still in the field).

    Args:
        frame: The ball rows written by ``fetch_scorecards.py``
            (``playhq_balls.csv``), with a ``fielder`` column.

    Returns:
        One row per player credited with at least one dismissal: catches,
        run outs, and the total, most credited first.
    """
    fielded = frame[frame["fielder"].fillna("") != ""]
    catches = fielded.loc[fielded["run_out"] == 0, "fielder"].value_counts()
    run_outs = fielded.loc[fielded["run_out"] == 1, "fielder"].value_counts()
    players = sorted(set(fielded["fielder"]))
    table = pd.DataFrame({
        "player": players,
        "catches": [int(catches.get(p, 0)) for p in players],
        "run_outs": [int(run_outs.get(p, 0)) for p in players],
    })
    table["total"] = table["catches"] + table["run_outs"]
    return table.sort_values("total", ascending=False).reset_index(drop=True)


def effect_table(idata, players: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Posterior mean and 90% interval of every player's effects.

    Args:
        idata: Posterior from ``fit_ball_model``.
        players: Restrict to these players; all when omitted.

    Returns:
        One row per player and effect, on the logit scale (positive means
        more of the outcome).
    """
    names = [str(p) for p in idata.posterior["bat_d"].coords["player"].values]
    keep = names if players is None else list(players)
    rows = []
    for role in ("bat", "bowl"):
        for c in COMPONENTS:
            draws = _flat(idata, f"{role}_{c}")
            for p in keep:
                d = draws[:, names.index(p)]
                lo, hi = np.percentile(d, [5, 95])
                rows.append({"player": p, "role": role, "component": c,
                             "effect": d.mean(), "lo": lo, "hi": hi})
    return pd.DataFrame(rows)
