#pragma once

#include <algorithm>
#include <cmath>

namespace ciru_mtp_cost_policy {

constexpr int kNativeMaxDepth = 7; // anchor + seven proposals remains in the T<=8 native path
constexpr double kFixedCostMs = 41.40575;
constexpr double kProposalCostMs = 10.97125;

// Beta(0.5,0.5) smoothed, with the two sparse middle bins pooled once.
inline double calibrated_probability(double top_k10_p) {
    if (!std::isfinite(top_k10_p) || top_k10_p < 0.0 || top_k10_p > 1.0) {
        return 7.5 / 26.0; // conservative finite fallback
    }
    if (top_k10_p < 0.5) {
        return 7.5 / 26.0;
    }
    if (top_k10_p < 0.75) {
        return 16.5 / 34.0;
    }
    if (top_k10_p < 0.97) {
        return 33.5 / 43.0;
    }
    return 359.5 / 377.0;
}

inline int effective_cap(int configured_n_max, int cycle_n_max) {
    if (configured_n_max < 1) {
        return 0;
    }
    int cap = std::min(configured_n_max, kNativeMaxDepth);
    if (cycle_n_max > 0) {
        cap = std::min(cap, cycle_n_max);
    }
    return cap;
}

inline double complete_cycle_cost_ms(int depth) {
    return kFixedCostMs + kProposalCostMs * depth;
}

struct ChainState {
    int depth = 0;
    double prefix_survival = 1.0;
    double expected_committed = 1.0; // target's own token
};

struct Decision {
    bool valid = false;
    bool stop = true;
    int best_depth = 0;
    double calibrated_p = 0.0;
    double prefix_survival = 0.0;
    double expected_committed = 0.0;
    double current_score = 0.0;
    double best_score = 0.0;
};

// Called exactly once after each kept proposal. The current confidence is the
// explicit forecast for each unseen future proposal within this cycle.
inline Decision after_kept_proposal(ChainState & state, double top_k10_p,
                                    int depth, int cap) {
    Decision result;
    if (depth != state.depth + 1 || depth < 1 || cap < depth || cap > kNativeMaxDepth ||
        !std::isfinite(state.prefix_survival) || !std::isfinite(state.expected_committed)) {
        return result; // fail closed: no further draft decode
    }

    const double p = calibrated_probability(top_k10_p);
    state.prefix_survival *= p;
    state.expected_committed += state.prefix_survival;
    state.depth = depth;
    if (!std::isfinite(state.prefix_survival) || !std::isfinite(state.expected_committed)) {
        return result;
    }

    result.valid = true;
    result.calibrated_p = p;
    result.prefix_survival = state.prefix_survival;
    result.expected_committed = state.expected_committed;
    result.current_score = state.expected_committed / complete_cycle_cost_ms(depth);
    result.best_score = result.current_score;
    result.best_depth = depth;

    double forecast_survival = state.prefix_survival;
    double forecast_committed = state.expected_committed;
    for (int future_depth = depth + 1; future_depth <= cap; ++future_depth) {
        forecast_survival *= p;
        forecast_committed += forecast_survival;
        const double score = forecast_committed / complete_cycle_cost_ms(future_depth);
        // Ties choose the shorter current/best depth.
        if (score > result.best_score + 1e-12) {
            result.best_score = score;
            result.best_depth = future_depth;
        }
    }
    result.stop = result.best_depth == depth;
    return result;
}

} // namespace ciru_mtp_cost_policy
