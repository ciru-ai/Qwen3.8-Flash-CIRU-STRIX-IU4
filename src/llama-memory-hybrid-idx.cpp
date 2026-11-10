#include "llama-memory-hybrid-idx.h"

#include <cstdlib>
#include "ggml-cpu.h"
#include "ggml-alloc.h"

#include "prefix.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iterator>
#include <map>
#include <stdexcept>

//
// llama_memory_hybrid_idx
//

// A qwen4exp MTP context is built with a recurrent filter that matches nothing (the nextn layer is not recurrent) so
// it can carry an indexer cache for sparse draft attention. Skip the recurrent child there: an empty recurrent cache
// still refuses partial seq_rm, which aborts the server on the first cache trim.
static bool hybrid_idx_no_recr(const llama_memory_recurrent * r) {
    if (!r) { return true; }
    for (ggml_tensor * t : r->r_l) { if (t) { return false; } }
    for (ggml_tensor * t : r->s_l) { if (t) { return false; } }
    return true;
}

// Experimental correctness fallback for a shared physical KV stream. Keep
// sequence-local QSA block/tail metadata while retaining pooled KV allocation.
// The default fixed-partition path is unchanged.
static bool qsa_unified_serial() {
    static const bool enabled = [] {
        const char * value = getenv("CIRU_QSA_UNIFIED_SERIAL");
        return value && atoi(value) != 0;
    }();
    return enabled;
}

llama_memory_hybrid_idx::llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
    const layer_filter_cb & filter_idx) :
    llama_memory_hybrid(
        model,
        type_k, type_v, v_trans, kv_size, n_pad, n_swa, swa_type,
        type_r, type_s, rs_size,
        n_seq_max, n_rs_seq, offload, unified,
        filter_attn, filter_recr),
    hparams_idx(model.hparams),
    mem_idx(filter_idx == nullptr ? nullptr : [&] {
        // MQA with a single key head of indexer_head_size, as llama_kv_cache_dsa shapes its own
        std::fill(hparams_idx.n_head_kv_arr.begin(), hparams_idx.n_head_kv_arr.end(), 1);
        hparams_idx.n_embd_head_k_full = model.hparams.indexer_head_size;

        // the cached indexer keys are raw, rotation happens after pooling at read time, so a
        // K-shift must not rotate them while the stream copies in the same update still apply
        hparams_idx.rope_type = LLAMA_ROPE_TYPE_NONE;

        // fool llama_kv_cache into thinking this is a MLA cache, so it won't cache V tensors
        hparams_idx.n_embd_head_k_mla_impl = model.hparams.indexer_head_size;
        hparams_idx.n_embd_head_v_mla_impl = model.hparams.indexer_head_size;

        LLAMA_LOG_INFO("%s: creating indexer KV cache, size = %u cells\n", __func__, kv_size);

        return new llama_kv_cache(
            model, hparams_idx, type_k, type_v, v_trans, offload, unified,
            kv_size, n_seq_max, n_pad, n_swa, swa_type,
            nullptr, filter_idx, nullptr, nullptr, "idx_");
    }()) {
#if defined(GGML_USE_HIP)
    const char * cache_env = getenv("CIRU_QSA_CACHE");
    incremental_qsa = cache_env && atoi(cache_env) != 0 && mem_idx && offload && n_swa == 0;
#endif
    if (!incremental_qsa) { return; }
    qsa_capacity = kv_size; qsa_row_used.assign((kv_size+3)/4, 0);
    LLAMA_LOG_INFO("%s: CIRU incremental QSA cache enabled, %u cells\n", __func__, kv_size);
    const int layers = model.hparams.n_layer_all;
    qsa_keys.resize(layers, nullptr);
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr> contexts;
    for (int il=0; il<layers; ++il) {
        if (!model.hparams.has_kv(il) || !filter_idx(il) || model.hparams.dsv4_compress_ratios[il] != 4) { continue; }
        auto * buft = ggml_backend_dev_buffer_type(model.dev_layer(il));
        auto & ctx = contexts[buft];
        if (!ctx) {
            ctx.reset(ggml_init({size_t(layers)*ggml_tensor_overhead(), nullptr, true}));
            if (!ctx) { throw std::runtime_error("QSA cache context allocation failed"); }
        }
        qsa_keys[il] = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, model.hparams.indexer_head_size, (kv_size+3)/4);
        ggml_format_name(qsa_keys[il], "cache_qsa_k_l%d", il);
    }
    for (auto & entry : contexts) {
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(entry.second.get(), entry.first);
        if (!buffer) { throw std::runtime_error("QSA cache buffer allocation failed"); }
        ggml_backend_buffer_clear(buffer, 0);
        qsa_buffers.emplace_back(std::move(entry.second), buffer);
    }
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // note: repeats llama_memory_hybrid::init_batch, as the indexer needs the attention slot infos that the base context hides
    const bool unified_serial = get_mem_attn()->get_n_stream() == 1 && qsa_unified_serial();

    // Pooled sequences on the incremental QSA path may share a ubatch. The first attempt groups them;
    // if the grouped layout cannot be proven safe the split is redone with one sequence per ubatch.
    std::vector<bool> groupable;
    // CIRU_QSA_BATCH_SCOPE: 1 = draft context only, 2 = target context only, 3 (default) = both
    static const int scope = getenv("CIRU_QSA_BATCH_SCOPE") ? atoi(getenv("CIRU_QSA_BATCH_SCOPE")) : 3;
    const bool in_scope = (scope & (hybrid_idx_no_recr(get_mem_recr()) ? 1 : 2)) != 0;
    const uint32_t group_max = unified_serial && !embd_all && in_scope ? qsa_batch_groupable(balloc, groupable) : 1;

    for (int attempt = group_max > 1 ? 0 : 1; attempt < 2; ++attempt) {
        const bool grouped = attempt == 0;

        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;
        bool multi = false;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all && !unified_serial) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (get_mem_attn()->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = get_mem_recr()->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0,
                        unified_serial ? (grouped ? group_max : 1) : 0, grouped ? &groupable : nullptr);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            multi = multi || ubatch.n_seqs > 1;
            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!hybrid_idx_no_recr(get_mem_recr()) && !get_mem_recr()->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = get_mem_attn()->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        if (grouped && multi && !qsa_batch_validate(ubatches, heads_attn)) {
            static unsigned logs = 0;
            if (logs++ < 8) { LLAMA_LOG_WARN("%s: grouped QSA split rejected, decoding the sequences one by one\n", __func__); }
            continue;
        }
        if (grouped && multi && getenv("CIRU_QSA_BATCH_TRACE")) {
            static unsigned logs = 0;
            if (logs++ < 400) {
                for (const auto & u : ubatches) {
                    fprintf(stderr, "CIRU_QSA_BATCH %s tokens=%u seqs=%u per_seq=%u first_seq=%d pos0=%d\n",
                            hybrid_idx_no_recr(get_mem_recr()) ? "draft" : "target", u.n_tokens, u.n_seqs, u.n_seq_tokens,
                            (int) u.seq_id[0][0], (int) u.pos[0]);
                }
            }
        }

        // the indexer uses the attention cache's slot layout; a separate one can drift from it
        llama_kv_cache::slot_info_vec_t heads_idx;
        if (mem_idx) {
            heads_idx = heads_attn;
        }

        return std::make_unique<llama_memory_hybrid_idx_context>(
                this, std::move(heads_attn), std::move(heads_idx), std::move(ubatches));
    }

    return std::make_unique<llama_memory_hybrid_idx_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_full() {
    return std::make_unique<llama_memory_hybrid_idx_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid_idx::init_update(llama_context * lctx, bool optimize) {
    auto result = std::make_unique<llama_memory_hybrid_idx_context>(this, lctx, optimize);
    if (result->get_status() != LLAMA_MEMORY_STATUS_NO_UPDATE) { qsa_invalidate(); }
    return result;
}

void llama_memory_hybrid_idx::clear(bool data) {
    qsa_seqs.clear(); std::fill(qsa_row_used.begin(), qsa_row_used.end(), 0);
    qsa_row_hint = 0; qsa_rows_in_use = 0; qsa_active.clear();
    if (data) { for (const auto & entry : qsa_buffers) { ggml_backend_buffer_clear(entry.second.get(), 0); } }
    llama_memory_hybrid::clear(data);

    if (mem_idx) {
        mem_idx->clear(data);
    }
}

bool llama_memory_hybrid_idx::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // same order as llama_memory_hybrid::seq_rm: the recurrent cache can refuse, so try it first
    if (!hybrid_idx_no_recr(get_mem_recr()) && !get_mem_recr()->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    if (incremental_qsa) {
        // only the named sequence is touched: the other sequences keep their cells, rows and readiness
        const auto trim = [&](qsa_seq_state & state) {
            if (!state.prefix.valid) { state.recover_pending = true; return; }
            if (p1 < 0 || size_t(p1) >= state.prefix.cells.size()) {
                state.prefix.truncate(std::max(0, p0));
                const size_t complete = state.prefix.cells.size()/4;
                for (auto & ready : state.ready) { ready = std::min<int64_t>(ready, complete); }
                qsa_release_rows(state, complete);
            } else { qsa_invalidate_seq(state); }
        };
        if (seq_id < 0) { for (auto & entry : qsa_seqs) { trim(entry.second); } }
        else { auto it = qsa_seqs.find(seq_id); if (it != qsa_seqs.end()) { trim(it->second); } }
    }
    if (mem_idx) {
        mem_idx->seq_rm(seq_id, p0, p1);
    }

    return get_mem_attn()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid_idx::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    qsa_invalidate();
    llama_memory_hybrid::seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (mem_idx) {
        mem_idx->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    }
}

void llama_memory_hybrid_idx::seq_keep(llama_seq_id seq_id) {
    for (auto & entry : qsa_seqs) { if (entry.first != seq_id) { qsa_invalidate_seq(entry.second); } }
    qsa_active.clear();
    llama_memory_hybrid::seq_keep(seq_id);

    if (mem_idx) {
        mem_idx->seq_keep(seq_id);
    }
}

void llama_memory_hybrid_idx::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    qsa_invalidate();
    llama_memory_hybrid::seq_add(seq_id, p0, p1, shift);

    if (mem_idx) {
        mem_idx->seq_add(seq_id, p0, p1, shift);
    }
}

void llama_memory_hybrid_idx::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    qsa_invalidate();
    llama_memory_hybrid::seq_div(seq_id, p0, p1, d);

    if (mem_idx) {
        mem_idx->seq_div(seq_id, p0, p1, d);
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid_idx::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = llama_memory_hybrid::memory_breakdown();

    if (mem_idx) {
        for (const auto & buft_size : mem_idx->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }

    for (const auto & entry : qsa_buffers) {
        mb[ggml_backend_buffer_get_type(entry.second.get())] += ggml_backend_buffer_get_size(entry.second.get());
    }
    return mb;
}

void llama_memory_hybrid_idx::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        get_mem_attn()->state_write(io, seq_id, flags);
    }
    if (!hybrid_idx_no_recr(get_mem_recr())) { get_mem_recr()->state_write(io, seq_id, flags); }

    // [TAG_HYBRID_IDX_STATE] the indexer section goes last, so it is a pure suffix: an old reader stops early instead of misparsing it
    // The indexer mirrors the attention cache, so it uses the same PARTIAL_ONLY gate.
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        if (mem_idx) {
            mem_idx->state_write(io, seq_id, flags);
        }
    }

}

void llama_memory_hybrid_idx::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) { qsa_invalidate(); }
    // note: repeats llama_memory_hybrid::state_read
    // the indexer needs the attention cache's cells, and a half-failed restore must leave all three caches alike

    // [TAG_HYBRID_IDX_SINFO]
    // the indexer restore adopts the attention cache's layout instead of searching for cells of its own
    // two find_slot calls agree only while both caches see the same occupancy, which a restore cannot promise
    llama_kv_cache::slot_info_vec_t sinfos_attn;

    try {
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            get_mem_attn()->state_read_sinfo(io, seq_id, flags, mem_idx ? &sinfos_attn : nullptr, nullptr);
        }

        if (!hybrid_idx_no_recr(get_mem_recr())) { get_mem_recr()->state_read(io, seq_id, flags); }

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
        if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
            if (mem_idx) {
                mem_idx->state_read_sinfo(io, seq_id, flags, nullptr, &sinfos_attn);
            }
        }

    } catch (...) {
        // a half-restored context is the one state the indexer cannot fix by itself: attention holds new cells, the indexer old ones
        // drop what was being restored from all of them, which is a state they do agree on.
        state_drop(seq_id);

        throw;
    }
}

void llama_memory_hybrid_idx::state_drop(llama_seq_id seq_id) {
    qsa_invalidate();
    // dropped directly, not via seq_rm: the recurrent cache may refuse it and then only the other two get cleared
    if (seq_id < 0) {
        clear(true);

        return;
    }

    get_mem_attn()->seq_rm(seq_id, -1, -1);
    if (!hybrid_idx_no_recr(get_mem_recr())) { get_mem_recr()->seq_rm(seq_id, -1, -1); }

    if (mem_idx) {
        mem_idx->seq_rm(seq_id, -1, -1);
    }
}

llama_kv_cache * llama_memory_hybrid_idx::get_mem_idx() const {
    return mem_idx.get();
}

void llama_memory_hybrid_idx::set_input_qsa(
        ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
        ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio, bool blk_bias) const {
    set_input_qsa_impl(cell_blk, blk_cells, blk_pos, bias, nullptr, ubatch, ratio, blk_bias);
}

void llama_memory_hybrid_idx::set_input_qsa_blocks(
        ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
        ggml_tensor * bias, ggml_tensor * tail_idxs, const llama_ubatch * ubatch, uint32_t ratio) const {
    set_input_qsa_impl(cell_blk, blk_cells, blk_pos, bias, tail_idxs, ubatch, ratio, true);
}

void llama_memory_hybrid_idx::set_input_qsa_impl(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        ggml_tensor * tail_idxs,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const {
    if (tail_idxs && cell_blk->ne[1] == 1 && qsa_metadata(blk_cells, blk_pos, bias, tail_idxs, *ubatch, ratio)) {
        return;
    }
    GGML_ASSERT(ratio > 0);
    GGML_ASSERT(get_mem_idx() != nullptr);

    GGML_ASSERT(tail_idxs || ggml_backend_buffer_is_host(cell_blk->buffer));

    const int64_t n_kv     = cell_blk->ne[0];
    const int64_t n_ns     = cell_blk->ne[1];        // streams in this ubatch
    const int64_t n_blocks = blk_pos->ne[0]/(4*n_ns);
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t r        = ratio;

    GGML_ASSERT(n_tokens % n_ns == 0);
    const int64_t n_tps = n_tokens/n_ns;             // tokens per stream

    int32_t * dst_cell_blk  = tail_idxs ? nullptr : (int32_t *) cell_blk->data;
    int32_t * dst_blk_cells = (int32_t *) blk_cells->data;
    int32_t * dst_blk_pos   = (int32_t *) blk_pos->data;
    const bool compact = bias->type == GGML_TYPE_I32;
    float * dst_bias = compact ? nullptr : (float *) bias->data;
    int32_t * limits = compact ? (int32_t *) bias->data : nullptr;
    if (compact) {
        GGML_ASSERT(tail_idxs && blk_bias && n_ns == 1 && ggml_nelements(bias) == n_blocks+n_tps);
        for (int64_t i=0;i<n_tokens;++i) { GGML_ASSERT(ubatch->seq_id[i][0] == ubatch->seq_id[0][0]); }
    }
    int32_t * dst_tail = tail_idxs ? (int32_t *) tail_idxs->data : nullptr;
    if (tail_idxs) {
        GGML_ASSERT(blk_bias && r > 1 && ggml_backend_buffer_is_host(tail_idxs->buffer));
        GGML_ASSERT(tail_idxs->ne[0] == r-1 && tail_idxs->ne[1] == n_tps && tail_idxs->ne[2] == n_ns);
        std::fill(dst_tail, dst_tail + ggml_nelements(tail_idxs), -1);
    }

    // a block is keyed on (sequence set, index bucket): a unified cache counts every sequence
    // from zero, so the bucket alone would pool two sequences into one block
    GGML_ASSERT(r <= 64);
    const uint64_t slots_full = r == 64 ? ~uint64_t(0) : ((uint64_t(1) << r) - 1);

    // TODO: this runs per ubatch and is O(n_kv) per stream, about 865 us at 33k context. the cost
    //       is the per-cell scan rather than these allocations, so hoisting them buys nothing
    std::vector<int32_t>  blk_of(n_kv);
    std::vector<int32_t>  cell_grp(n_kv);
    std::vector<int32_t>  grp_head(n_blocks);
    std::map<int64_t, int32_t> overflow_head;
    std::vector<int32_t>  grp_next;
    std::vector<int32_t>  grp_first;
    std::vector<int32_t>  grp_slot0;
    std::vector<uint64_t> grp_slots;
    std::vector<int32_t>  grp_bid;
    std::vector<int32_t>  bid_idx;
    std::vector<int32_t>  bid_cell;
    std::vector<int32_t>  bid_slot0;

    std::vector<int32_t> order;
    std::vector<int32_t> rank;

    std::fill(dst_blk_pos, dst_blk_pos + 4*n_blocks*n_ns, 0);

    for (int64_t s = 0; s < n_ns; ++s) {
        // ubatch index s*n_tps belongs to this stream; ask which cells array it uses
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = get_mem_idx()->get_cells(seq_of_stream);
        bool single_seq = true;
        for (int64_t i=0; i<n_tokens; ++i) { single_seq = single_seq && ubatch->seq_id[i][0] == seq_of_stream; }
        // One primary sequence per ubatch: shared-prefix cells still belong to it and must be included.
        // Grouped ubatches use qsa_metadata; a mixed ubatch reaching this point keeps the sequence-aware grouping.
        const bool isolate_seq = n_ns == 1 && qsa_unified_serial() && single_seq;

        int32_t * cur_cell_blk  = dst_cell_blk ? dst_cell_blk + s*n_kv : nullptr;
        int32_t * cur_blk_cells = dst_blk_cells + s*(r*n_blocks);

        std::fill(cur_blk_cells, cur_blk_cells + r*n_blocks, 0);

        bid_idx  .clear();
        bid_cell .clear();
        bid_slot0.clear();

        int n_seq_present = 0;

        for (int sq = 0; sq < LLAMA_MAX_SEQ && n_seq_present < 2; ++sq) {
            if (cells.seq_pos_min(sq) >= 0) {
                n_seq_present++;
            }
        }

        const bool one_seq = isolate_seq || n_seq_present <= 1;

        bool dup = false;

        bool ranked = false;

        auto group_cells = [&]() {
            // -1 means no usable block: an incomplete or short group cannot be pooled
            std::fill(blk_of.begin(),   blk_of.end(),   -1);
            std::fill(cell_grp.begin(), cell_grp.end(), -1);
            std::fill(grp_head.begin(), grp_head.end(), -1);

            grp_next .clear();
            grp_first.clear();
            grp_slot0.clear();
            grp_slots.clear();
            grp_bid  .clear();

            overflow_head.clear();
            dup = false;

            for (int64_t j = 0; j < n_kv; ++j) {
                if (cells.is_empty(j) || (isolate_seq && !cells.seq_has(j, seq_of_stream))) {
                    continue;
                }

                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);
                const int64_t pb  = idx/r;

                // MTP skips image rows but keeps text positions. Logical buckets can exceed the cell window.
                GGML_ASSERT(idx >= 0);
                int32_t & head = pb < n_blocks ? grp_head[pb] : overflow_head.emplace(pb, -1).first->second;
                int32_t g = -1;

                for (int32_t c = head; c >= 0; c = grp_next[c]) {
                    if (one_seq || cells.seq_get_all((uint32_t) grp_first[c]) == cells.seq_get_all((uint32_t) j)) {
                        g = c;
                        break;
                    }
                }

                if (g < 0) {
                    g = (int32_t) grp_first.size();

                    grp_next .push_back(head);
                    grp_first.push_back((int32_t) j);
                    grp_slot0.push_back(-1);
                    grp_slots.push_back(0);
                    grp_bid  .push_back(-1);

                    head = g;
                }

                const uint64_t bit = uint64_t(1) << (idx%r);

                dup |= (grp_slots[g] & bit) != 0;

                cell_grp[j]   = g;
                grp_slots[g] |= bit;

                if (idx%r == 0) {
                    grp_slot0[g] = (int32_t) j;
                }
            }
        };

        group_cells();

        // mrope repeats one position across an image, so rank cells instead of using the position
        if (dup && ubatch->is_pos_2d() && one_seq) {
            order.clear();
            order.reserve(n_kv);

            for (int64_t j = 0; j < n_kv; ++j) {
                if (!cells.is_empty(j) && (!isolate_seq || cells.seq_has(j, seq_of_stream))) {
                    order.push_back((int32_t) j);
                }
            }

            // same total order the mrope causal mask uses: pos, then ext.y, then ext.x
            std::sort(order.begin(), order.end(), [&cells](int32_t a, int32_t b) {
                const llama_pos pa = cells.pos_get(a);
                const llama_pos pb = cells.pos_get(b);

                if (pa != pb) {
                    return pa < pb;
                }

                const auto & ea = cells.ext_get(a);

                return cells.ext_get(b).is_2d_gt(ea.x, ea.y);
            });

            rank.assign(n_kv, -1);

            for (int64_t k = 0; k < (int64_t) order.size(); ++k) {
                rank[order[k]] = (int32_t) k;
            }

            ranked = true;

            group_cells();
        }

        int32_t n_bid = 0;

        auto emit_bucket = [&](int64_t pb, int32_t head) {
            for (int32_t g = head; g >= 0; g = grp_next[g]) {
                if (grp_slots[g] != slots_full) {
                    continue;
                }

                grp_bid[g] = n_bid++;

                bid_idx  .push_back((int32_t) (pb*r));
                bid_cell .push_back(grp_first[g]);
                bid_slot0.push_back(grp_slot0[g]);
            }
        };

        for (int64_t pb = 0; pb < n_blocks; ++pb) {
            emit_bucket(pb, grp_head[pb]);
        }
        for (const auto & entry : overflow_head) {
            emit_bucket(entry.first, entry.second);
        }

        GGML_ASSERT(n_bid <= n_blocks);

        for (int32_t b = 0; b < n_bid; ++b) {
            int32_t sec_pos[4] = { bid_idx[b], bid_idx[b], bid_idx[b], bid_idx[b] };

            if (ranked) {
                const int32_t   c = bid_slot0[b];
                const llama_pos p = cells.pos_get(c);
                const auto &    e = cells.ext_get(c);

                sec_pos[0] = p;
                sec_pos[1] = e.y;
                sec_pos[2] = e.x;
                sec_pos[3] = p;
            }

            for (int64_t sec = 0; sec < 4; ++sec) {
                dst_blk_pos[sec*(n_blocks*n_ns) + s*n_blocks + b] = sec_pos[sec];
            }
        }

        // unpooled cells all point at one spare block. a spare block exists only when some
        // cell is unpooled: n_bid == n_blocks means every cell sits in a full block.
        const bool     have_dead = n_bid < n_blocks;
        const int32_t  dead_bid  = have_dead ? n_bid : n_blocks - 1;

        for (int64_t j = 0; j < n_kv; ++j) {
            const int32_t g = cell_grp[j];

            blk_of[j] = g < 0 ? -1 : grp_bid[g];

            if (blk_of[j] >= 0) {
                const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                cur_blk_cells[blk_of[j]*r + (idx%r)] = (int32_t) j;
            }

            if (cur_cell_blk) { cur_cell_blk[j] = blk_of[j] < 0 ? dead_bid : blk_of[j]; }
        }

        std::vector<int32_t> group_members;
        if (dst_tail) {
            group_members.assign(grp_first.size()*r, -1);
            for (int64_t j=0;j<n_kv;++j) {
                const int32_t g=cell_grp[j];
                if (g<0) { continue; }
                const int64_t idx=ranked ? rank[j] : cells.pos_get(j);
                const int64_t slot=g*r+idx%r;
                GGML_ASSERT(group_members[slot]<0 || group_members[slot]==j);
                group_members[slot]=(int32_t)j;
            }
        }

        if (compact) {
            const llama_seq_id seq = ubatch->seq_id[0][0];
            for (int64_t b=0;b<n_blocks;++b) {
                limits[b] = b<n_bid && cells.seq_has((uint32_t)bid_cell[b],seq) ? bid_idx[b] : INT32_MAX;
            }
        }

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t      i      = s*n_tps + ii;
            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            int64_t q = ubatch->pos[i];

            if (ranked) {
                const llama_pos qt = ubatch->pos[i];
                const llama_pos qy = ubatch->pos[i + n_tokens];
                const llama_pos qx = ubatch->pos[i + n_tokens*2];

                int64_t lo = 0;
                int64_t hi = (int64_t) order.size();

                while (lo < hi) {
                    const int64_t   mid = (lo + hi)/2;
                    const int32_t   c   = order[mid];
                    const llama_pos pc  = cells.pos_get(c);

                    if (pc < qt || (pc == qt && !cells.ext_get(c).is_2d_gt(qx, qy))) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }

                q = lo - 1;
            }

            // the tail is an incomplete block and is always visible, as in the reference
            const int64_t tail_start = (q + 1)/r*r;
            if (dst_tail && q+1>tail_start) {
                const int64_t pb=tail_start/r;
                const auto overflow = overflow_head.find(pb);
                const int32_t head = pb >= 0 && pb < n_blocks ? grp_head[pb] :
                    (overflow != overflow_head.end() ? overflow->second : -1);
                if (head >= 0) {
                    int32_t * tail=dst_tail+(s*n_tps+ii)*(r-1);
                    for (int32_t g=head;g>=0;g=grp_next[g]) {
                        if (!cells.seq_has((uint32_t)grp_first[g],seq_id)) { continue; }
                        for (int64_t slot=0;slot<q+1-tail_start;++slot) {
                            const int32_t cell=group_members[g*r+slot];
                            if (cell<0 || !cells.seq_has((uint32_t)cell,seq_id)) { continue; }
                            GGML_ASSERT(tail[slot]<0 || tail[slot]==cell);
                            tail[slot]=cell;
                        }
                    }
                }
            }

            if (compact) {
                GGML_ASSERT(tail_start >= 0 && tail_start <= 16777216);
                limits[n_blocks+ii] = (int32_t)tail_start;
                continue;
            }

            if (blk_bias) {
                // a block sits wholly inside or outside the tail, so one value covers it
                // the caller adds the attention mask, which drops empty, foreign and future cells
                float * cur_blk_bias = dst_bias + i*n_blocks;

                for (int64_t b = 0; b < n_blocks; ++b) {
                    if (b >= n_bid || !cells.seq_has((uint32_t) bid_cell[b], seq_id)) {
                        cur_blk_bias[b] = -INFINITY;
                        continue;
                    }

                    // finite, so it can never meet a -inf and produce a nan
                    cur_blk_bias[b] = bid_idx[b] >= tail_start ? (dst_tail ? -INFINITY : 1e9f) : 0.0f;
                }

                // the spare block holds the unpooled cells, which are the incomplete tail, so
                // it gets the tail value. it must stay finite: a sequence with fewer than
                // `ratio` cells owns no full block, and a row of -inf only gives a nan.
                if (have_dead) {
                    cur_blk_bias[dead_bid] = dst_tail ? -INFINITY : 1e9f;
                }

                continue;
            }

            float * cur_bias = dst_bias + i*n_kv;

            for (int64_t j = 0; j < n_kv; ++j) {
                float v = -INFINITY;

                if (!cells.is_empty(j) && cells.seq_has(j, seq_id)) {
                    const int64_t idx = ranked ? rank[j] : cells.pos_get(j);

                    if (idx <= q) {
                        // finite, so it can never meet a -inf and produce a nan
                        v = idx >= tail_start ? 1e9f : (blk_of[j] < 0 ? -INFINITY : 0.0f);
                    }
                }

                cur_bias[j] = v;
            }
        }
    }
}

//
// llama_memory_hybrid_idx_context
//

// streams in each ubatch's slot info, matching get_k/get_v's `ns`
static std::vector<uint32_t> llama_memory_hybrid_idx_ns(const llama_kv_cache::slot_info_vec_t & sinfos) {
    std::vector<uint32_t> res;
    res.reserve(sinfos.size());

    for (const auto & sinfo : sinfos) {
        res.push_back(sinfo.s1 - sinfo.s0 + 1);
    }

    return res;
}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_status status) :
    llama_memory_hybrid_context(status) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem) :
    llama_memory_hybrid_context(mem),
    mem(mem),
    // graph reservation walks a full context, and qwen4exp builds the sparse attention only when this is set
    // without it the reserved worst case is the dense graph, so ggml-alloc must grow the buffer on the first decode
    ns_ubatch(mem->get_mem_idx() == nullptr ?
        std::vector<uint32_t>() : std::vector<uint32_t>{ mem->get_mem_idx()->get_n_stream() }),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx())) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                  llama_context * lctx,
                           bool   optimize) :
    llama_memory_hybrid_context(mem, lctx, optimize),
    mem(mem),
    // update() applies a pending cross-stream seq_cp, else the copy keeps stale indexer keys
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        mem->get_mem_idx()->init_update(lctx, optimize)) {}

llama_memory_hybrid_idx_context::llama_memory_hybrid_idx_context(
        llama_memory_hybrid_idx * mem,
                slot_info_vec_t   sinfos_attn,
                slot_info_vec_t   sinfos_idx,
      std::vector<llama_ubatch>   ubatches) :
    // note: the base copies the ubatches; ctx_idx gets a copy of its own
    llama_memory_hybrid_context(mem, std::move(sinfos_attn), ubatches),
    mem(mem),
    ns_ubatch(llama_memory_hybrid_idx_ns(sinfos_idx)),
    qsa_slots(sinfos_idx),
    ctx_idx(mem->get_mem_idx() == nullptr ? nullptr :
        new llama_kv_cache_context(mem->get_mem_idx(), std::move(sinfos_idx), ubatches)) {}

bool llama_memory_hybrid_idx_context::next() {
    if (ctx_idx) {
        ctx_idx->next();
    }

    ++i_cur;

    return llama_memory_hybrid_context::next();
}

bool llama_memory_hybrid_idx_context::apply() {
    bool res = llama_memory_hybrid_context::apply();

    if (ctx_idx) {
        res = res & ctx_idx->apply();
        if (res && i_cur < qsa_slots.size()) {
            const_cast<llama_memory_hybrid_idx *>(mem)->qsa_apply(ctx_idx->get_ubatch(), qsa_slots[i_cur]);
        } else if (!res) { const_cast<llama_memory_hybrid_idx *>(mem)->qsa_invalidate(); }
    }

    return res;
}

const llama_kv_cache_context * llama_memory_hybrid_idx_context::get_idx() const {
    return static_cast<const llama_kv_cache_context *>(ctx_idx.get());
}

uint32_t llama_memory_hybrid_idx_context::get_n_stream() const {
    GGML_ASSERT(i_cur < ns_ubatch.size());

    return ns_ubatch[i_cur];
}

void llama_memory_hybrid_idx_context::set_input_qsa(
        ggml_tensor * cell_blk,
        ggml_tensor * blk_cells,
        ggml_tensor * blk_pos,
        ggml_tensor * bias,
        const llama_ubatch * ubatch,
        uint32_t ratio,
        bool blk_bias) const {
    GGML_ASSERT(mem != nullptr);

    mem->set_input_qsa(cell_blk, blk_cells, blk_pos, bias, ubatch, ratio, blk_bias);
}

void llama_memory_hybrid_idx_context::set_input_qsa_blocks(
        ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
        ggml_tensor * bias, ggml_tensor * tail_idxs, const llama_ubatch * ubatch, uint32_t ratio) const {
    GGML_ASSERT(mem != nullptr);
    mem->set_input_qsa_blocks(cell_blk, blk_cells, blk_pos, bias, tail_idxs, ubatch, ratio);
}

bool llama_memory_hybrid_idx_context::qsa_position_prefix(const llama_ubatch & ubatch) const {
    if (qsa_prefix_matches(ubatch)) { return true; }
    if (!qsa_scalar_visibility(ubatch)) { return false; }
    const llama_seq_id seq=ubatch.seq_id[0][0];
    return qsa_single_sequence_prefix(mem->get_mem_idx()->get_cells(seq),get_idx()->get_n_kv(),seq);
}

int64_t llama_memory_hybrid_idx_context::qsa_sequence_cells(const llama_ubatch & ubatch) const {
    // Full-cache graph reservation has no committed slot metadata. Preserve its
    // worst-case sparse graph rather than sizing workspace from an empty cache.
    if (!mem || qsa_slots.empty() || !get_idx() || get_n_stream() != 1 ||
            !qsa_unified_serial() || !ubatch.n_tokens || !ubatch.seq_id || !ubatch.n_seq_id ||
            ubatch.n_seq_id[0] < 1 || !ubatch.seq_id[0]) { return -1; }
    const llama_seq_id seq = ubatch.seq_id[0][0];
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id[i] < 1 || !ubatch.seq_id[i] || ubatch.seq_id[i][0] != seq) { return -1; }
    }
    const auto & cells = mem->get_mem_idx()->get_cells(seq);
    int64_t count = 0;
    for (uint32_t j = 0; j < get_idx()->get_n_kv(); ++j) {
        if (!cells.is_empty(j) && cells.seq_has(j, seq)) { ++count; }
    }
    return count;
}

bool llama_memory_hybrid_idx_context::qsa_scalar_visibility(const llama_ubatch & ubatch) const {
    if (qsa_prefix_matches(ubatch)) { return true; }
    // see qwen4exp_qsa_embd_ok: a batch with tokens AND embeddings is the MTP draft head, not a vision batch
    const char * embd_env = getenv("LLAMA_QSA_TOKEN_EMBD");
    const bool embd_ok = !ubatch.embd || !embd_env || atoi(embd_env) != 0;
    if (get_n_stream()!=1 || !get_idx() || !ubatch.token || !embd_ok || !ubatch.pos || !ubatch.n_tokens ||
            !ubatch.n_pos || !ubatch.seq_id || !ubatch.n_seq_id) { return false; }
    if (ubatch.n_seq_id[0]<1 || !ubatch.seq_id[0]) { return false; }
    const llama_seq_id seq=ubatch.seq_id[0][0];
    for (uint32_t i=0;i<ubatch.n_tokens;++i) {
        if (ubatch.n_seq_id[i]<1 || !ubatch.seq_id[i] || ubatch.seq_id[i][0]!=seq ||
                ubatch.pos[i]<0 || ubatch.pos[i]>=16777216) { return false; }
        for (uint32_t axis=1;axis<ubatch.n_pos;++axis) {
            if (ubatch.pos[i+axis*ubatch.n_tokens]!=ubatch.pos[i]) { return false; }
        }
    }
    if (ubatch.is_pos_2d()) {
        const auto & cells=mem->get_mem_idx()->get_cells(seq);
        for (uint32_t j=0;j<get_idx()->get_n_kv();++j) {
            if (!cells.is_empty(j) && cells.seq_has(j,seq) && cells.ext_get(j).is_2d_gt(cells.pos_get(j),cells.pos_get(j))) { return false; }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------
// Incremental QSA key cache, one state per sequence.
//
// The per-layer key cache is shared: every complete block of every tracked sequence owns one row.
// The graph scores all rows and the compact visibility limit hides the rows of other sequences
// (INT32_MAX), so a pooled multi-slot server keeps the incremental path while slots alternate.
// A lone sequence gets rows 0..n-1, which is exactly the former single-sequence layout.
// ---------------------------------------------------------------------------------------------

llama_memory_hybrid_idx::qsa_seq_state & llama_memory_hybrid_idx::qsa_state(llama_seq_id seq) {
    auto it = qsa_seqs.find(seq);
    if (it == qsa_seqs.end()) {
        it = qsa_seqs.emplace(seq, qsa_seq_state()).first;
        it->second.prefix = qsa_prefix_state(qsa_capacity);
        // the sequence may already own cells (restore, earlier untracked batches): rebuild from the cache
        it->second.prefix.valid = false;
        it->second.recover_pending = true;
        it->second.ready.assign(qsa_keys.size(), 0);
    }
    return it->second;
}

const llama_memory_hybrid_idx::qsa_seq_state * llama_memory_hybrid_idx::qsa_group_state(size_t group) const {
    if (group >= qsa_active.size()) { return nullptr; }
    const auto it = qsa_seqs.find(qsa_active[group]);
    return it != qsa_seqs.end() && it->second.prefix.valid ? &it->second : nullptr;
}

static int qsa_fast_max_tokens() {
    static const int max_tokens = [] {
        const char * value = getenv("CIRU_QSA_INDEXER_MAX");
        return value ? std::max(0, std::min(512, atoi(value))) : 8;
    }();
    return max_tokens;
}

// CIRU_QSA_BATCH_SEQS=N lets up to N pooled sequences share one ubatch (0/1 = one sequence per ubatch).
static uint32_t qsa_batch_max() {
    static const uint32_t value = [] {
        const char * env = getenv("CIRU_QSA_BATCH_SEQS");
        return env ? uint32_t(std::max(1, std::min(16, atoi(env)))) : 1u;
    }();
    return value;
}

// Which sequences of this batch may share a ubatch: tracked, appending a short run of text tokens,
// and long enough that they take the sparse path on their own. Returns the group size limit (1 = none).
uint32_t llama_memory_hybrid_idx::qsa_batch_groupable(const llama_batch_allocr & balloc, std::vector<bool> & groupable) const {
    if (qsa_batch_max() <= 1 || !incremental_qsa || !mem_idx || mem_idx->get_n_stream() != 1) { return 1; }
    const llama_batch & b = balloc.get_batch();
    if (!b.token || !b.pos || !b.seq_id || !b.n_seq_id || b.n_tokens < 2) { return 1; }
    std::map<llama_seq_id, std::pair<int32_t, llama_pos>> runs; // seq -> (tokens, first position)
    for (int32_t i = 0; i < b.n_tokens; ++i) {
        if (b.n_seq_id[i] != 1 || !b.seq_id[i]) { return 1; }
        auto it = runs.find(b.seq_id[i][0]);
        if (it == runs.end()) { runs.emplace(b.seq_id[i][0], std::make_pair(1, b.pos[i])); }
        else { ++it->second.first; }
    }
    if (runs.size() < 2) { return 1; }
    groupable.assign(LLAMA_MAX_SEQ, false);
    uint32_t count = 0;
    for (const auto & run : runs) {
        const auto it = qsa_seqs.find(run.first);
        if (run.first < 0 || run.first >= LLAMA_MAX_SEQ || it == qsa_seqs.end()) { continue; }
        const auto & state = it->second;
        if (!state.prefix.valid || state.recover_pending || run.second.first > qsa_fast_max_tokens() ||
            size_t(run.second.second) != state.prefix.cells.size() ||
            state.prefix.cells.size() + run.second.first <= size_t(hparams_idx.indexer_top_k) + 8) { continue; }
        groupable[run.first] = true; ++count;
    }
    return count >= 2 ? std::min(count, qsa_batch_max()) : 1;
}

// Dry run of qsa_apply for every grouped ubatch: true only if each sequence stays on the incremental path.
bool llama_memory_hybrid_idx::qsa_batch_validate(const std::vector<llama_ubatch> & ubatches,
        const llama_kv_cache::slot_info_vec_t & sinfos) const {
    if (ubatches.size() != sinfos.size()) { return false; }
    std::map<llama_seq_id, size_t> sizes; // simulated prefix length
    size_t new_rows = 0;
    int32_t max_row = -1;
    uint32_t extent = 0;
    for (size_t n = 0; n < ubatches.size(); ++n) {
        const auto & u = ubatches[n];
        if (u.n_seqs <= 1) {
            // a sequence must not appear in both a grouped and a later single ubatch state we did not simulate
            if (u.seq_id && u.n_tokens && u.seq_id[0]) { sizes.erase(u.seq_id[0][0]); }
            continue;
        }
        const auto & slots = sinfos[n];
        if (!u.token || !u.pos || !u.n_pos || !u.seq_id || !u.n_seq_id || !u.equal_seqs() ||
            u.n_seq_tokens*u.n_seqs != u.n_tokens || slots.n_stream() != 1 || slots.size() != u.n_tokens ||
            int(u.n_seq_tokens) > qsa_fast_max_tokens() || u.n_tokens >= 128) { return false; }
        for (uint32_t g = 0; g < u.n_seqs; ++g) {
            const uint32_t first = g*u.n_seq_tokens;
            if (u.n_seq_id[first] != 1 || !u.seq_id[first]) { return false; }
            const llama_seq_id seq = u.seq_id[first][0];
            const auto it = qsa_seqs.find(seq);
            if (it == qsa_seqs.end() || !it->second.prefix.valid || it->second.recover_pending) { return false; }
            const auto & state = it->second;
            auto size = sizes.find(seq);
            if (size == sizes.end()) { size = sizes.emplace(seq, state.prefix.cells.size()).first; }
            else if (size->second == SIZE_MAX) { return false; }
            if (size->second != state.prefix.cells.size()) { return false; } // one grouped ubatch per sequence
            for (uint32_t i = 0; i < u.n_seq_tokens; ++i) {
                const uint32_t idx = first+i;
                if (u.n_seq_id[idx] != 1 || !u.seq_id[idx] || u.seq_id[idx][0] != seq ||
                    int64_t(u.pos[idx]) != int64_t(size->second)+i) { return false; }
                for (uint32_t axis = 1; axis < u.n_pos; ++axis) {
                    if (u.pos[idx+axis*u.n_tokens] != u.pos[idx]) { return false; }
                }
                const uint32_t cell = slots.idxs[0][idx];
                if (cell >= state.prefix.positions.size() || state.prefix.positions[cell] >= 0) { return false; }
                extent = std::max(extent, cell+1);
            }
            for (uint32_t h = 0; h < g; ++h) { if (u.seq_id[h*u.n_seq_tokens][0] == seq) { return false; } }
            const size_t after = size->second + u.n_seq_tokens;
            if (after <= size_t(hparams_idx.indexer_top_k)) { return false; } // would be the dense dispatch alone
            new_rows += after/4 - state.rows.size();
            max_row = std::max(max_row, state.max_row);
            size->second = SIZE_MAX;
            extent = std::max(extent, mem_idx->get_cells(seq).used_max_p1());
        }
    }
    if (qsa_rows_in_use + new_rows > qsa_row_used.size()) { return false; }
    // lowest-free allocation: a new row index is below the row count after the allocation
    max_row = std::max<int64_t>(max_row, int64_t(qsa_rows_in_use + new_rows) - 1);
    return max_row < int32_t((extent+3)/4);
}

void llama_memory_hybrid_idx::qsa_release_rows(qsa_seq_state & state, size_t keep) {
    bool lost_max = false;
    for (size_t b = keep; b < state.rows.size(); ++b) {
        const int32_t row = state.rows[b];
        qsa_row_used[row] = 0; --qsa_rows_in_use;
        qsa_row_hint = std::min(qsa_row_hint, size_t(row));
        if (size_t(row) != b) { --state.displaced; }
        if (row == state.max_row) { lost_max = true; }
    }
    if (keep < state.rows.size()) { state.rows.resize(keep); }
    if (state.rows.empty()) { state.max_row = -1; }
    else if (lost_max) { state.max_row = *std::max_element(state.rows.begin(), state.rows.end()); }
}

bool llama_memory_hybrid_idx::qsa_grow_rows(qsa_seq_state & state) {
    const size_t complete = state.prefix.cells.size()/4;
    while (state.rows.size() < complete) {
        while (qsa_row_hint < qsa_row_used.size() && qsa_row_used[qsa_row_hint]) { ++qsa_row_hint; }
        if (qsa_row_hint >= qsa_row_used.size()) { return false; }
        const int32_t row = int32_t(qsa_row_hint);
        qsa_row_used[row] = 1; ++qsa_rows_in_use;
        if (size_t(row) != state.rows.size()) { ++state.displaced; }
        state.max_row = std::max(state.max_row, row);
        state.rows.push_back(row);
    }
    return true;
}

void llama_memory_hybrid_idx::qsa_invalidate_seq(qsa_seq_state & state) {
    state.prefix.invalidate(); state.recover_pending = true;
    std::fill(state.ready.begin(), state.ready.end(), 0);
    qsa_release_rows(state, 0);
}

void llama_memory_hybrid_idx::qsa_invalidate() {
    for (auto & entry : qsa_seqs) { qsa_invalidate_seq(entry.second); }
    qsa_active.clear();
}

bool llama_memory_hybrid_idx::qsa_recover(llama_seq_id seq, qsa_seq_state & state) {
    if (mem_idx->get_n_stream() != 1) { return false; }
    const auto & raw = mem_idx->get_cells(seq);
    if (raw.size() != qsa_capacity) { return false; }
    // other sequences may share the stream: collect only this sequence's cells
    std::vector<std::pair<int32_t, uint32_t>> own;
    for (uint32_t cell=raw.used_min(); cell<raw.used_max_p1(); ++cell) {
        if (raw.is_empty(cell) || !raw.seq_has(cell, seq)) { continue; }
        const int32_t pos = raw.pos_get(cell);
        const auto & ext = raw.ext_get(cell);
        if (pos < 0 || raw.seq_get_all(cell).count() != 1 ||
            !((ext.x == 0 && ext.y == 0) || (ext.x == pos && ext.y == pos))) { return false; }
        own.emplace_back(pos, cell);
    }
    qsa_prefix_state rebuilt(qsa_capacity);
    rebuilt.cells.assign(own.size(), -1);
    for (const auto & entry : own) {
        if (size_t(entry.first) >= own.size() || rebuilt.cells[entry.first] >= 0) { return false; }
        rebuilt.cells[entry.first] = entry.second; rebuilt.positions[entry.second] = entry.first;
    }
    rebuilt.sequence = seq;
    for (size_t b=0; b<rebuilt.cells.size()/4; ++b) { rebuilt.block_positions.push_back(b*4); }
    qsa_release_rows(state, 0);
    std::fill(state.ready.begin(), state.ready.end(), 0);
    state.prefix = std::move(rebuilt);
    return true;
}

void llama_memory_hybrid_idx::qsa_apply(const llama_ubatch & u, const llama_kv_cache::slot_info & slots) {
    if (!incremental_qsa) { return; }
    qsa_active.clear();
    // an untracked write: forget every sequence this ubatch names (all of them if it names none)
    const auto reject = [&]() {
        bool named = false;
        if (u.seq_id && u.n_seq_id) {
            for (uint32_t i=0; i<u.n_tokens; ++i) {
                for (int32_t k=0; u.seq_id[i] && k<u.n_seq_id[i]; ++k) {
                    named = true;
                    const auto it = qsa_seqs.find(u.seq_id[i][k]);
                    if (it != qsa_seqs.end()) { qsa_invalidate_seq(it->second); it->second.recover_pending = false; }
                    else { auto & fresh = qsa_state(u.seq_id[i][k]); fresh.recover_pending = false; }
                }
            }
        }
        if (!named) { qsa_invalidate(); for (auto & entry : qsa_seqs) { entry.second.recover_pending = false; } }
    };
    if (!u.token || !u.pos || !u.n_tokens || slots.n_stream() != 1 ||
        slots.size() != u.n_tokens || !u.n_pos || !u.seq_id || !u.n_seq_id || u.n_seq_id[0] != 1 || !u.seq_id[0]) {
        reject(); return;
    }
    // a grouped ubatch holds n_seqs runs of n_seq_tokens tokens, one sequence per run
    const uint32_t groups = u.n_seqs > 1 && u.equal_seqs() && u.n_seq_tokens*u.n_seqs == u.n_tokens ? u.n_seqs : 1;
    const uint32_t per = u.n_tokens/groups;
    for (uint32_t g=0; g<groups; ++g) {
        const uint32_t first = g*per;
        if (u.n_seq_id[first] != 1 || !u.seq_id[first]) { reject(); return; }
        const auto seq = u.seq_id[first][0];
        for (uint32_t h=0; h<g; ++h) { if (u.seq_id[h*per][0] == seq) { reject(); return; } }
        for (uint32_t i=0; i<per; ++i) {
            const uint32_t idx = first+i;
            if (u.n_seq_id[idx] != 1 || !u.seq_id[idx] || u.seq_id[idx][0] != seq || int64_t(u.pos[idx]) != int64_t(u.pos[first])+i) {
                reject(); return;
            }
            for (uint32_t axis=1; axis<u.n_pos; ++axis) {
                if (u.pos[idx+axis*u.n_tokens] != u.pos[idx]) { reject(); return; }
            }
        }
    }
    std::vector<llama_seq_id> applied;
    bool ok = true;
    for (uint32_t g=0; g<groups; ++g) {
        const uint32_t first = g*per;
        const auto seq = u.seq_id[first][0];
        auto & state = qsa_state(seq);
        if (!state.prefix.valid && (!state.recover_pending || !qsa_recover(seq, state))) { state.recover_pending = false; ok = false; continue; }
        state.recover_pending = false;
        const bool grown = groups == 1
            ? state.prefix.apply(seq, u.pos[first], slots.idxs[0])
            : state.prefix.apply(seq, u.pos[first], std::vector<uint32_t>(slots.idxs[0].begin()+first, slots.idxs[0].begin()+first+per));
        if (!grown || !qsa_grow_rows(state)) {
            qsa_invalidate_seq(state); state.recover_pending = false; ok = false; continue;
        }
        // every row must stay inside the block count the graph derives from the pool extent
        const auto & raw = mem_idx->get_cells(seq);
        if (state.max_row >= int32_t((raw.used_max_p1()+3)/4)) { qsa_invalidate_seq(state); ok = false; continue; }
        applied.push_back(seq);
    }
    if (ok) { qsa_active = std::move(applied); }
    else if (groups > 1) {
        LLAMA_LOG_ERROR("%s: a grouped ubatch left the incremental QSA path (%u sequences)\n", __func__, groups);
    }
}

bool llama_memory_hybrid_idx::qsa_prefix_matches(const llama_ubatch & u) const {
    if (!incremental_qsa || !u.token || !u.pos || !u.n_tokens || !u.n_pos || !u.seq_id || !u.n_seq_id ||
        qsa_active.empty() || u.n_tokens % qsa_active.size() != 0) { return false; }
    const uint32_t groups = qsa_active.size(), per = u.n_tokens/groups;
    if (groups > 1 && (u.n_seqs != groups || u.n_seq_tokens != per)) { return false; }
    for (uint32_t g=0; g<groups; ++g) {
        const auto * state = qsa_group_state(g);
        const uint32_t first = g*per;
        if (!state || u.pos[first] != state->prefix.begin || int64_t(per) != state->prefix.end-state->prefix.begin) { return false; }
        for (uint32_t i=0; i<per; ++i) {
            const uint32_t idx = first+i;
            if (u.n_seq_id[idx] != 1 || !u.seq_id[idx] || u.seq_id[idx][0] != qsa_active[g] || u.pos[idx] != u.pos[first]+int32_t(i)) { return false; }
            for (uint32_t a=1; a<u.n_pos; ++a) { if (u.pos[idx+a*u.n_tokens] != u.pos[idx]) { return false; } }
        }
        // Long ubatches use bounded and packed scorers that assume row == block for the whole cache.
        // Keep them on the generic path unless this sequence is alone in the identity layout.
        if (u.n_tokens >= 128 && (groups != 1 || state->displaced != 0 || qsa_rows_in_use != state->rows.size())) { return false; }
    }
    return true;
}

size_t llama_memory_hybrid_idx::qsa_groups(const llama_ubatch & u) const {
    return qsa_active.size() > 1 && qsa_prefix_matches(u) ? qsa_active.size() : 1;
}

bool llama_memory_hybrid_idx::qsa_fast(int il, const llama_ubatch & u) const {
    const int max_tokens = qsa_fast_max_tokens();
    if (!qsa_prefix_matches(u)) { return false; }
    const uint32_t groups = qsa_active.size(), per = u.n_tokens/groups;
    bool fast = int(per) <= max_tokens && qsa_keys.at(il);
    for (uint32_t g=0; fast && g<groups; ++g) {
        const auto * state = qsa_group_state(g);
        fast = state->ready.at(il) >= int64_t(state->prefix.previous_size/4) &&
            state->prefix.cells.size()/4 >= (per+3)/4+1;
    }
    static unsigned trace_count = 0;
    if (getenv("CIRU_QSA_INDEXER_TRACE") && trace_count++ < 96) {
        const auto * state = qsa_group_state(0);
        fprintf(stderr, "CIRU_INDEXER_PATH seq=%d groups=%u layer=%d tokens=%u fast=%d limit=%d ready=%lld previous=%zu total=%zu rows_in_use=%zu displaced=%zu\n",
                (int) qsa_active[0], groups, il, u.n_tokens, int(fast), max_tokens, (long long) state->ready.at(il),
                state->prefix.previous_size/4, state->prefix.cells.size()/4, qsa_rows_in_use, state->displaced);
    }
    return fast;
}

ggml_tensor * llama_memory_hybrid_idx::qsa_cache(ggml_context * ctx, int il, int64_t blocks) const {
    if (!incremental_qsa || !qsa_keys.at(il)) { return nullptr; }
    auto * k = qsa_keys[il]; GGML_ASSERT(blocks <= k->ne[1]);
    return ggml_view_2d(ctx, k, k->ne[0], blocks, k->nb[1], 0);
}

void llama_memory_hybrid_idx::qsa_fill_updates(ggml_tensor * members, ggml_tensor * pos, ggml_tensor * rows) const {
    const int groups = qsa_active.size(); GGML_ASSERT(groups >= 1 && rows->ne[0] % groups == 0);
    const int total = rows->ne[0], count = total/groups;
    auto * c = (int32_t *) members->data; auto * p = (int32_t *) pos->data; auto * w = (int64_t *) rows->data;
    std::vector<int32_t> ids;
    for (int g=0; g<groups; ++g) {
        const auto * state = qsa_group_state(g); GGML_ASSERT(state);
        const auto & prefix = state->prefix;
        const int complete = prefix.cells.size()/4;
        const int first = prefix.begin/4, last = std::min(complete, (prefix.end+3)/4);
        GGML_ASSERT(last-first <= count && complete >= count && int(state->rows.size()) == complete);
        ids.clear();
        for (int b=first; b<last; ++b) { ids.push_back(b); }
        for (int b=0; int(ids.size())<count; ++b) { if (b<first || b>=last) { ids.push_back(b); } }
        for (int i=0; i<count; ++i) {
            const int o = g*count+i;
            w[o] = state->rows[ids[i]];
            for (int j=0; j<4; ++j) { c[4*o+j] = prefix.cells[ids[i]*4+j]; p[j*total+o] = ids[i]*4; }
        }
    }
}

void llama_memory_hybrid_idx::qsa_commit(int il) const {
    GGML_ASSERT(!qsa_active.empty());
    for (size_t g=0; g<qsa_active.size(); ++g) {
        const auto * state = qsa_group_state(g); GGML_ASSERT(state);
        state->ready.at(il) = state->prefix.cells.size()/4;
    }
}

bool llama_memory_hybrid_idx::qsa_metadata(ggml_tensor * members, ggml_tensor * pos, ggml_tensor * bias,
        ggml_tensor * tails, const llama_ubatch & u, uint32_t ratio) const {
    if (ratio != 4 || bias->type != GGML_TYPE_I32 || !qsa_prefix_matches(u)) { return false; }
    const size_t groups = qsa_active.size(), per = u.n_tokens/groups;
    const size_t blocks = pos->ne[0]/4;
    // limits: one visibility row per group (block start position, INT32_MAX = hidden), then one tail start per token
    if (size_t(ggml_nelements(bias)) != groups*blocks + u.n_tokens) { return false; }
    std::vector<const qsa_seq_state *> active(groups);
    for (size_t g=0; g<groups; ++g) {
        active[g] = qsa_group_state(g); GGML_ASSERT(active[g]);
        GGML_ASSERT(active[g]->max_row < int32_t(blocks));
    }
    auto * c = (int32_t *) members->data; auto * p = (int32_t *) pos->data;
    auto * lim = (int32_t *) bias->data; auto * tail = (int32_t *) tails->data;
    GGML_ASSERT(c && lim && tail);
    if (groups == 1 && qsa_rows_in_use == active[0]->rows.size() && active[0]->displaced == 0) {
        // lone sequence in the identity layout: the former contiguous fill
        const size_t complete = active[0]->rows.size();
        std::copy_n(active[0]->prefix.cells.data(), complete*4, c);
        std::fill(c+complete*4, c+blocks*4, 0);
        if (p) {
            for (int a=0; a<4; ++a) {
                std::copy_n(active[0]->prefix.block_positions.data(), complete, p+a*blocks);
                std::fill(p+a*blocks+complete, p+(a+1)*blocks, 0);
            }
        }
        std::copy_n(active[0]->prefix.block_positions.data(), complete, lim);
        std::fill(lim+complete, lim+blocks, INT32_MAX);
    } else {
        // shared rows: describe every tracked row, expose to each group only its own sequence's blocks
        std::fill(c, c+blocks*4, 0);
        if (p) { std::fill(p, p+blocks*4, 0); }
        std::fill(lim, lim+groups*blocks, INT32_MAX);
        for (const auto & entry : qsa_seqs) {
            const auto & state = entry.second;
            if (!state.prefix.valid) { continue; }
            int group = -1;
            for (size_t g=0; g<groups; ++g) { if (active[g] == &state) { group = int(g); } }
            for (size_t b=0; b<state.rows.size(); ++b) {
                const size_t row = state.rows[b];
                if (row >= blocks) { continue; }
                std::copy_n(state.prefix.cells.data()+b*4, 4, c+row*4);
                const int32_t position = state.prefix.block_positions[b];
                if (p) { for (int a=0; a<4; ++a) { p[a*blocks+row] = position; } }
                if (group >= 0) { lim[size_t(group)*blocks+row] = position; }
            }
        }
    }
    std::fill(tail, tail+3*u.n_tokens, -1);
    for (size_t g=0; g<groups; ++g) {
        for (size_t i=0; i<per; ++i) {
            const size_t idx = g*per+i;
            int32_t end = u.pos[idx]+1, start = end/4*4; lim[groups*blocks+idx] = start;
            for (int j=0; j<end-start; ++j) { tail[3*idx+j] = active[g]->prefix.cells[start+j]; }
        }
    }
    return true;
}

std::vector<int64_t> llama_memory_hybrid_idx::qsa_own_widths(const llama_ubatch & u) const {
    static const bool enabled = getenv("CIRU_QSA_OWN_ROWS") && atoi(getenv("CIRU_QSA_OWN_ROWS")) != 0;
    if (!enabled || u.n_tokens >= 128 || !qsa_prefix_matches(u)) { return {}; }
    for (size_t il = 0; il < qsa_keys.size(); ++il) {
        if (qsa_keys[il] && !qsa_fast(int(il), u)) { return {}; } // a refill pass needs the shared-row metadata
    }
    const size_t groups = qsa_active.size();
    const auto * first = qsa_group_state(0);
    // a lone sequence in the identity layout already scores exactly its own rows
    if (groups == 1 && qsa_rows_in_use == first->rows.size() && first->displaced == 0) { return {}; }
    const int64_t pad = 256, budget = (int64_t(hparams_idx.indexer_top_k)/4 + pad - 1)/pad*pad;
    std::vector<int64_t> widths;
    for (size_t g = 0; g < groups; ++g) {
        const auto * state = qsa_group_state(g);
        const int64_t complete = state->prefix.cells.size()/4;
        if (int64_t(state->rows.size()) != complete) { return {}; }
        widths.push_back(std::max(budget, (complete + pad - 1)/pad*pad));
    }
    return widths;
}

void llama_memory_hybrid_idx::qsa_fill_own(ggml_tensor * rows, ggml_tensor * cells, ggml_tensor * positions,
        ggml_tensor * limits, ggml_tensor * tails, const llama_ubatch & u) const {
    const size_t groups = qsa_active.size(), per = u.n_tokens/groups;
    const size_t total = rows->ne[0];
    GGML_ASSERT(size_t(cells->ne[0]) == 4*total && size_t(positions->ne[0]) == total && size_t(limits->ne[0]) == total + u.n_tokens);
    auto * w = (int32_t *) rows->data; auto * c = (int32_t *) cells->data; auto * p = (int32_t *) positions->data;
    auto * lim = (int32_t *) limits->data; auto * tail = (int32_t *) tails->data;
    GGML_ASSERT(w && c && p && lim && tail);
    std::fill(w, w+total, 0); std::fill(c, c+4*total, 0); std::fill(p, p+total, 0); std::fill(lim, lim+total, INT32_MAX);
    std::fill(tail, tail+3*u.n_tokens, -1);
    size_t off = 0;
    const int64_t pad = 256, budget = (int64_t(hparams_idx.indexer_top_k)/4 + pad - 1)/pad*pad;
    for (size_t g = 0; g < groups; ++g) {
        const auto * state = qsa_group_state(g); GGML_ASSERT(state);
        const size_t complete = state->prefix.cells.size()/4;
        const size_t width = std::max<int64_t>(budget, (int64_t(complete) + pad - 1)/pad*pad);
        GGML_ASSERT(off + width <= total && state->rows.size() == complete);
        std::copy_n(state->rows.data(), complete, w+off);
        std::copy_n(state->prefix.cells.data(), complete*4, c+off*4);
        std::copy_n(state->prefix.block_positions.data(), complete, p+off);
        std::copy_n(state->prefix.block_positions.data(), complete, lim+off);
        for (size_t i = 0; i < per; ++i) {
            const size_t idx = g*per+i;
            int32_t end = u.pos[idx]+1, start = end/4*4; lim[total+idx] = start;
            for (int j = 0; j < end-start; ++j) { tail[3*idx+j] = state->prefix.cells[start+j]; }
        }
        off += width;
    }
    GGML_ASSERT(off == total);
}
