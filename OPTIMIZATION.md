# Optimizing Flash runtime dispatch

The small-batch C1 change lets eligible AMD batches of 9-16 rows use the existing R4 expert kernels. These joint batches previously selected padded grouped matrix kernels. One predicate controls both grouped bypass and vector admission, preserving the original geometry, layout, stride and alignment checks. The kernel arithmetic and model weights are unchanged. The dispatcher selects this route automatically when the existing R4 backend and tensor requirements apply. The normal default remains one slot.

A retained 16-row complete expert-operation comparison measured 4.701845 -> 1.536682ms, or 3.06x. This includes packing, projections, activation, weighting and reduction. It is a component result, not a 3x model-speed claim. One concurrent-serving observation also improved, but generated outputs and request trajectories differed, so it does not establish an equal-output or repeatable causal ratio.

Numerical checks remain scoped. The eight-row control was exact; the 16-row component had relative L2 error about 5.49e-6. A limited four-sequence full-model check preserved 16/16 top choices, with mean KL about 0.0001973 and maximum total variation about 2.76%. Twelve positions were predictable JSON endings. These checks do not establish broad task-quality or stochastic-acceptance equivalence. Floating-point differences should be evaluated against useful quality and stability, while storage, indexing, routing and sequence state must remain correct.

To find additional improvements:

1. Inventory actual prefill, decode and speculative phases, including row counts, tensor dimensions/strides, expert occupancy and active dispatches.
2. Inspect dispatch boundaries and fallback conditions. A configured optimization is useful only when the measured workload actually takes that path.
3. Rank complete costs with correct units and overlap. Include packing, copies, routing and launch overhead. A component occupying fraction f of serial work and accelerated by s offers at most the planning ratio 1/(1-f+f/s).
4. Test one mechanism on captured inputs and a representative resident weight bank. Prove execution, check numerical/state behavior, then measure complete clean work.
5. Advance a useful component result to matched serving under unchanged model, context, sampling and speculation settings. Keep patches independent and retain negative or inconclusive results.

A separate experiment applied cooperative decoding to the small 32-row prefill expert tile. It preserved all 104,857,600 tested BF16 outputs, but complete GPU-stage time changed only 37.4404 -> 37.3025ms, a 0.37% reduction below its 2% screen threshold. That historical 16384-row stage result is practical parity, so the experiment is parked and excluded from the source upgrade. A shorter native prefill may have a different expert mix; that requires representative input evidence, not repetition of the same fixture.

New-package functional checks are distinct from performance experiments. They verify build/library identity, profile resolution, concurrent completion and preserved sequence behavior. A package smoke test does not create a new speed ratio or expand the numerical and workload limits above.
