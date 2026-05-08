import numpy as np
from scipy.stats import trim_mean
from typing import Literal

from config import Config
from audio_loader import load_audio
from loudness import extract_loudness
from spectral import extract_spectral
from stereo import extract_stereo
from temporal import extract_temporal


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features_from_audio(audio: np.ndarray, sr: int, config: Config) -> dict:
    features = {}
    features.update(extract_loudness(audio, sr, config))
    features.update(extract_spectral(audio, sr))
    features.update(extract_stereo(audio))
    features.update(extract_temporal(audio, sr))
    return features


def extract_features_from_file(file_path: str, config: Config) -> dict:
    audio, sr = load_audio(file_path, sr=config.sr)
    return extract_features_from_audio(audio, sr, config)


# ── Averaging ─────────────────────────────────────────────────────────────────

def average_features(feature_list: list[dict], config: Config) -> dict:
    avg = {}
    for key in feature_list[0]:
        values = [f[key] for f in feature_list if np.isfinite(f[key])]
        if not values:
            avg[key] = 0.0
            continue
        if config.avg_method == "mean":
            avg[key] = float(np.mean(values))
        elif config.avg_method == "median":
            avg[key] = float(np.median(values))
        else:
            avg[key] = float(trim_mean(values, 0.1))
    return avg


def std_features(feature_list: list[dict]) -> dict:
    """Per-feature standard deviation across references."""
    std = {}
    for key in feature_list[0]:
        values = [f[key] for f in feature_list if np.isfinite(f[key])]
        std[key] = float(np.std(values)) if len(values) > 1 else 0.0
    return std


# ── Raw comparison ────────────────────────────────────────────────────────────

def compare_features(reference: dict, user: dict) -> dict:
    return {
        key: {
            "reference":  reference[key],
            "user":       user[key],
            "difference": user[key] - reference[key],
        }
        for key in reference
    }


# ── Z-score normalisation ─────────────────────────────────────────────────────

def zscore_comparison(
    ref_avg: dict,
    ref_std: dict,
    user_features: dict,
) -> dict:
    zscores = {}
    for key in ref_avg:
        diff = user_features[key] - ref_avg[key]
        std  = ref_std.get(key, 0.0)
        if std > 1e-9:
            zscores[key] = diff / std
        else:
            scale = max(abs(ref_avg[key]), 1e-6)
            zscores[key] = diff / scale
    return zscores


# ── Deterministic interpretation ──────────────────────────────────────────────

_SEVERITY_LABELS = {
    "ignore":   "on target",
    "mild":     "mild",
    "moderate": "moderate",
    "strong":   "strong",
}

def _severity(z: float, config: Config) -> tuple[str, str]:
    """Return (severity_key, direction)."""
    direction = "higher" if z > 0 else "lower"
    az = abs(z)
    if az < config.z_ignore:
        return "ignore", direction
    if az < config.z_mild:
        return "mild", direction
    if az < config.z_moderate:
        return "moderate", direction
    return "strong", direction

def _domain_mild_count(features, severities):
    return sum(1 for f in features if severities[f][0] == "mild")


# ── Perceptual domain rules ───────────────────────────────────────────────────

def _rule_loudness(zscores: dict, sev: dict) -> list[str]:
    advice = []
    int_sev, int_dir     = sev["integrated_lufs"]
    mom_sev, mom_dir     = sev["momentary_lufs"]
    crest_sev, crest_dir = sev["crest_factor_db"]

    mild_count = _domain_mild_count(
        ["integrated_lufs", "momentary_lufs", "crest_factor_db"], sev
    )

    if int_sev == "mild" and int_dir == "lower":
        advice.append(
            "Integrated loudness is slightly below reference. A small gain increase (~0.5–1 dB) may help alignment."
        )

    elif int_sev in ("moderate", "strong") and int_dir == "lower":
        advice.append(
            "Integrated loudness is significantly below reference. Increase overall gain before the limiter by 1–3 dB, then re-limit."
        )

    if mom_sev in ("mild", "moderate", "strong") and mom_dir == "lower":
        advice.append(
            "Momentary loudness is slightly lower than reference. Check limiter behavior or mix density."
        )

    if crest_sev in ("mild", "moderate", "strong") and crest_dir == "higher":
        advice.append(
            "Crest factor is slightly high. Mild compression may improve consistency."
        )

    if mild_count >= 2 and not advice:
        advice.append(
            "Small loudness-related differences detected across multiple metrics. Minor gain staging or compression tweaks may improve alignment."
        )

    return advice


def _rule_brightness(zscores: dict, sev: dict) -> list[str]:
    advice = []
    c_sev, c_dir = sev["spectral_centroid"]
    r_sev, r_dir = sev["spectral_rolloff"]
    h_sev, h_dir = sev["high_energy_ratio"]
    m_sev, m_dir = sev["mid_energy_ratio"]

    if (c_sev not in ("ignore",) and c_dir == "higher"
            and r_sev not in ("ignore",) and r_dir == "higher"):
        if c_sev in ("moderate", "strong") or h_sev in ("moderate", "strong"):
            advice.append(
                "Brightness is elevated across both centroid and rolloff. "
                "Check cymbal levels and vocal high-end first (source fix). "
                "A gentle high-shelf cut at 8–10 kHz (–1 to –2 dB) on "
                "individual channels is preferred over a master bus shelf. "
                "If needed, a narrow dynamic EQ at 8–12 kHz on the master bus "
                "(threshold 1–2 dB GR) handles transient harshness only."
            )
        else:
            advice.append(
                "Slight brightness increase. Small high-shelf cut (–0.5 dB) "
                "at 10 kHz on the master bus is fine, but verify at source first."
            )

    elif (c_sev not in ("ignore",) and c_dir == "higher"
          and r_sev in ("ignore",)):
        if m_sev not in ("ignore",) and m_dir == "higher":
            advice.append(
                "Spectral centroid is high but rolloff is normal — this points "
                "to midrange congestion rather than excessive brightness. "
                "Look for masking between 1–4 kHz: vocals, guitars, synths, "
                "and snare presence often compete here. Surgical EQ cuts at "
                "the source (1–3 dB narrow notch per element) will clear "
                "perceived mud without thinning the mix."
            )

    elif (c_sev in ("moderate", "strong") and c_dir == "lower"
          and h_sev in ("moderate", "strong") and h_dir == "lower"):
        advice.append(
            "Mix is noticeably dark relative to references. Check whether "
            "high-frequency content was cut at source (cymbals, room mics, "
            "vocal air). A high-shelf boost at 10–12 kHz (+0.5–1.5 dB) on "
            "the master bus is acceptable for mild correction; larger moves "
            "should happen at the source."
        )

    return advice


def _rule_tonal_balance(zscores: dict, sev: dict) -> list[str]:
    advice = []

    l_sev, l_dir = sev["low_energy_ratio"]
    m_sev, m_dir = sev["mid_energy_ratio"]
    h_sev, h_dir = sev["high_energy_ratio"]

    mild_count = _domain_mild_count(
        ["low_energy_ratio", "mid_energy_ratio", "high_energy_ratio"], sev
    )

    if l_sev in ("mild", "moderate", "strong") and l_dir == "higher":
        advice.append(
            "Low-end energy is slightly elevated. Check kick/bass balance and low-mid buildup."
        )

    if m_sev in ("mild", "moderate", "strong") and m_dir == "higher":
        advice.append(
            "Midrange is slightly forward. Consider small EQ cuts around 500 Hz–2 kHz."
        )

    if mild_count >= 2 and not advice:
        advice.append(
            "Subtle tonal balance differences detected across bands. Minor EQ adjustments may improve overall balance."
        )

    return advice


def _rule_stereo(zscores: dict, sev: dict) -> list[str]:
    advice = []

    w_sev, w_dir = sev["stereo_width"]
    c_sev, c_dir = sev["correlation"]

    mild_count = _domain_mild_count(
        ["stereo_width", "correlation", "mid_side_ratio"], sev
    )

    if w_sev in ("mild", "moderate", "strong") and w_dir == "lower":
        advice.append(
            "Stereo width is slightly narrower than reference. Consider subtle widening at the arrangement level."
        )

    if c_sev in ("mild", "moderate", "strong") and c_dir == "lower":
        advice.append(
            "Stereo correlation suggests mild phase issues or over-widening. Check mono compatibility."
        )

    if mild_count >= 2 and not advice:
        advice.append(
            "Small stereo field differences detected. Minor spatial adjustments may improve width and balance."
        )

    return advice


def _rule_transients(zscores: dict, sev: dict) -> list[str]:
    advice = []

    td_sev, td_dir       = sev["transient_density"]
    rms_v_sev, rms_v_dir = sev["rms_variance"]

    mild_count = _domain_mild_count(
        ["transient_density", "rms_variance"], sev
    )

    if td_sev in ("mild", "moderate", "strong") and td_dir == "lower":
        advice.append(
            "Transient activity is slightly reduced. Consider easing compression or enhancing attack."
        )

    if rms_v_sev in ("mild", "moderate", "strong") and rms_v_dir == "lower":
        advice.append(
            "Dynamic variation is slightly reduced. The mix may benefit from less bus compression."
        )

    if mild_count >= 2 and not advice:
        advice.append(
            "Subtle transient and dynamic differences detected. Minor adjustments may improve punch and movement."
        )

    return advice


# ── Confidence scoring ────────────────────────────────────────────────────────

def reference_confidence(ref_std: dict, ref_list_len: int, config: Config) -> dict:
    confidence = {}
    for key, std in ref_std.items():
        if ref_list_len < 2:
            confidence[key] = "low"
        elif ref_list_len < config.min_refs_for_high_confidence:
            confidence[key] = "medium"
        else:
            confidence[key] = "high" if std < 0.2 * (abs(std) + 1e-9) * 5 else "medium"
    return confidence


def _confidence_from_std(ref_std: dict, ref_avg: dict, n_refs: int, config: Config) -> dict:
    confidence = {}
    for key in ref_avg:
        if n_refs < 2:
            confidence[key] = "low"
            continue
        scale = max(abs(ref_avg[key]), 1e-9)
        cv = ref_std.get(key, 0.0) / scale
        if n_refs >= config.min_refs_for_high_confidence and cv < 0.15:
            confidence[key] = "high"
        elif cv < 0.35:
            confidence[key] = "medium"
        else:
            confidence[key] = "low"
    return confidence


# ── Master interpreter ────────────────────────────────────────────────────────

def interpret(
    ref_avg: dict,
    ref_std: dict,
    user_features: dict,
    n_refs: int,
    config: Config,
) -> dict:

    zscores    = zscore_comparison(ref_avg, ref_std, user_features)
    severities = {k: _severity(z, config) for k, z in zscores.items()}
    confidence = _confidence_from_std(ref_std, ref_avg, n_refs, config)

    advice = {
        "Loudness & Dynamic Range": _rule_loudness(zscores, severities),
        "Tonal Balance & Brightness": (
            _rule_brightness(zscores, severities)
            + _rule_tonal_balance(zscores, severities)
        ),
        "Transients & Punch": _rule_transients(zscores, severities),
        "Stereo Image": _rule_stereo(zscores, severities),
    }

    # ── Priority scoring ──────────────────────────────────────────────

    def _domain_score(domain_name):
        s = 0
        s += len(advice[domain_name]) * 2
        for feature, (sev, _) in severities.items():
            if domain_name == "Loudness & Dynamic Range" and feature in [
                "integrated_lufs", "momentary_lufs", "crest_factor_db"
            ]:
                if sev == "mild":
                    s += 1
                elif sev == "moderate":
                    s += 2
                elif sev == "strong":
                    s += 3
            elif domain_name == "Tonal Balance & Brightness" and feature in [
                "spectral_centroid", "spectral_rolloff",
                "low_energy_ratio", "mid_energy_ratio", "high_energy_ratio"
            ]:
                if sev == "mild":
                    s += 1
                elif sev == "moderate":
                    s += 2
                elif sev == "strong":
                    s += 3
            elif domain_name == "Transients & Punch" and feature in [
                "transient_density", "rms_variance", "crest_factor_db"
            ]:
                if sev == "mild":
                    s += 1
                elif sev == "moderate":
                    s += 2
                elif sev == "strong":
                    s += 3
            elif domain_name == "Stereo Image" and feature in [
                "stereo_width", "correlation", "mid_side_ratio"
            ]:
                if sev == "mild":
                    s += 1
                elif sev == "moderate":
                    s += 2
                elif sev == "strong":
                    s += 3
        return s

    priority_order = sorted(
        advice.keys(),
        key=lambda d: _domain_score(d),
        reverse=True,
    )

    return {
        "zscores":        zscores,
        "severities":     severities,
        "advice":         advice,
        "confidence":     confidence,
        "priority_order": priority_order,
    }


# ── LLM prompt builder (phrasing only — no new decisions) ────────────────────

def build_llm_phrasing_prompt(interpretation: dict, ref_avg: dict, user_features: dict) -> str:
    lines = [
        "You are a senior professional mix and mastering engineer. Your job is to "
        "rephrase the structured mixing advice below into clear, direct, in-depth "
        "professional feedback — the kind a seasoned engineer would give in a session.\n"
        "\n"
        "For each point:\n"
        "  1. Explain what the problem is and why it matters perceptually.\n"
        "  2. Give a practical, specific action the engineer should take — name "
        "     the tool, the frequency, the amount, or the signal chain stage to check.\n"
        "  3. If relevant, mention what to listen for to confirm the fix is working.\n"
        "\n"
        "Do NOT invent new problems. Do NOT remove any point. Do NOT change the priority order. "
        "Do NOT generalise or pad with vague advice.\n",

        "## Structured advice to rephrase:\n",
    ]

    for domain in interpretation["priority_order"]:
        domain_advice = interpretation["advice"][domain]
        if not domain_advice:
            continue
        lines.append(f"### {domain}")
        for point in domain_advice:
            lines.append(f"- {point}")
        lines.append("")

    lines.append(
        "\n## Priority Order (do not change):\n"
        + "\n".join(
            f"{i+1}. {d}"
            for i, d in enumerate(interpretation["priority_order"])
            if interpretation["advice"][d]
        )
    )

    # Build a plain-English snapshot of the key numeric deltas for the summary
    key_features = {
        "integrated_lufs":  "integrated loudness",
        "spectral_centroid": "spectral centroid",
        "stereo_width":      "stereo width",
        "crest_factor_db":   "crest factor",
        "transient_density": "transient density",
    }
    delta_lines = []
    for feat, label in key_features.items():
        if feat in ref_avg and feat in user_features:
            delta = user_features[feat] - ref_avg[feat]
            direction = "above" if delta > 0 else "below"
            delta_lines.append(
                f"- {label}: user is {abs(delta):.2f} units {direction} the reference average"
            )

    lines.append(
        "\n## Reference vs User — Key Deltas (for your summary):\n"
        + "\n".join(delta_lines)
    )

    lines.append(
        "\n## Final instructions:\n"
        "Format your full response in markdown.\n"
        "Be concise and technically precise in each point.\n"
        "After covering all domains, write a ## Overall Picture section: "
        "2–3 sentences that summarise the character of the user's mix versus "
        "the references as a whole — what kind of mix it sounds like it is, "
        "where it sits tonally and dynamically, and the single most impactful "
        "change the engineer should make first.\n"
        "End with a ## Priority Order section listing the domains in the order given above."
        "Do NOT infer numerical values beyond what is explicitly stated. "
        "Do NOT exaggerate severity. If advice says 'slight' or 'mild', keep that scale.\n"
    )

    return "\n".join(lines)