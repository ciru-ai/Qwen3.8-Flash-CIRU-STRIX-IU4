#pragma once
#include "llama.h"

// Private opt-in prototype; one MTP sequence per process. No target logits
// or target weights are modified by this interface.
extern "C" LLAMA_API int ciru_mtp_shortlist_size(void);
extern "C" LLAMA_API void ciru_mtp_shortlist_update(
        const llama_model * draft, const float * target_logits, int n_vocab, llama_token last);
