# Flash v6.1 runtime notices

The runtime is derived from llama.cpp and the HaloBox Strix Halo release b1064. The source for HaloBox b1064 is commit b9f3c0e878f82f80ea1528969f305017a909b73e. The source patch, build instructions, and source integrity manifest are published with this release. The MIT license and original source notices apply.

Ciru’s compact C1 expert format is combined with work from the HaloBox, Gufo, and llama.cpp projects. The Qwen model and derived GGUF files have their own Qwen Community License 1.0, shipped with the model. See the model card for the pinned weights and PLE source.

The ROCm dependency installer retrieves HaloBox’s pinned original ROCm package and checks its SHA-256. Refer to its original release and AMD’s notices for the ROCm runtime components.
