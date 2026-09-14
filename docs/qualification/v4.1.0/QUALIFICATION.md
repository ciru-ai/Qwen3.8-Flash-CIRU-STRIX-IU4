# CIRU runtime v4.1.0 validation

The IO32 CPU-pager change preserves model weights, GPU arithmetic, page validation and ordered cache commits. Two 160MiB component outputs matched R2 byte for byte. Balanced native PP16384 improved 4.44% (955.916 to998.394 tokens/s), saving0.730 seconds. This native comparison used16 threads and batch/ubatch16384.

The IO32-only MTP6 quick screen used four A/C/C/A server loads,4k/16k prompts,256 generated tokens and seed123, for8 measured requests plus4 excluded warmups. All four pairs returned identical output tokens. TG was0.39% to1.62% higher. The serving configuration retained262144 context, batch/ubatch8192,8 threads,F16 target/draft KV,one slot,1GiB host cache and4GiB PLE cache. It is a bounded regression screen, not a universal TG guarantee. Minimum available RAM was7.413GiB.

The earlier IO32 cold recall, exact cached replay and cold16k completion checks passed. V4.0 quality, capacity and vision results remain under ../v4.0.0 in their original scope; they are not newly scored v4.1 results. Orca-specific IO32 performance was not measured. The rejected direct-Q5 expansion experiment is not included.

Final-package checks cover the actual shipped IO32 libraries and launcher/profile/UI, plus both variants' file selection and guards. Their release receipt accompanies the assets. The tested binary archive targets the recorded NixOS/ROCm10 environment; other systems build the matching source.

Halogen creator Peonist.ai originated the fast-prefill breakthrough. pwilkin supplied the open-source implementation; CIRU provides compatibility integration and validation.
