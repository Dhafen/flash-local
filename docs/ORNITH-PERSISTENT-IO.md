# Persistent expert I/O candidate

Isolated branch `codex/ornith-persistent-io`, native implementation source
`f705c8ceef722ad35c66554c657471271a34edf6`. This scheduling candidate is
**off by default and not included in the API-125 delivery**. No iPhone speed
or trained-model quality result is established.

The accepted phone-123 page-cache profile spends approximately 128–133 ms in
expert installation per 233–245 ms decode forward. Those spans can overlap
other work. The pinned upstream implementation creates and joins helper
threads for each packed expert wave. The candidate retains the same byte
ranges, slots and actual selected experts while reusing a bounded context-owned
pool. Its caller participates: N readers require N−1 persistent helpers.
Waves admit at most 256 jobs and 1–8 readers. Errors drain active work before
propagation; nested/concurrent waves fail. The owner must await decode before
teardown, and closing its context joins the helpers.

`flo_options.persistent_io` explicitly opts in. The existing path is unchanged
when false. New counters report actual helper starts, waves, jobs and live
helpers; configuration records the requested readers per context. The cached
legacy environment setting is not used to configure the new pool. Existing
legacy thread-count caching still needs attention before timing a sweep.
The patch manifest also hashes the pool header, preventing stale generated
source when only that header changes.

## Completed host controls

[Native control run 39223188783](https://github.com/Dhafen/flash-local/actions/runs/39223188783)
passes all nine combinations of F32, mixed Q4_K/Q6_K and 256-expert/top-eight
fixtures with 1, 4 and 8 readers. Each pooled execution closes before replaying
the identical streaming configuration with the legacy schedule. All complete
logit arrays and canonical routing/expert-set hashes must match bitwise;
actual pool use and owner teardown are checked. These are generated untrained
fixtures, not the installed Ornith weights.

| Reader configuration | F32 pairs | Mixed pairs | Wide top-eight pairs |
| --- | ---: | ---: | ---: |
| 1 | 25 | 25 | 43 |
| 4 | 25 | 25 | 43 |
| 8 | 25 | 25 | 43 |

All 279 paired context executions pass. Their maximum resident-reference
logit error is also zero, with complete layer checkpoints, reset replay and
callback-free controls retained. I/O job counts can differ with prefetch/cache
timing; this does not change the selected experts or resulting logits. This
control does not establish equal cache conditions for a speed comparison.

The same run passes 26 selected Swift controls and validates isolated headers,
symbols and imports for Mac, iPhone and Simulator. The actual Metal row controls
pass; complete hosted native Metal projections remain unavailable. CPU graph
comparisons do not substitute for a phone GPU check.

[Address/undefined-behavior sanitizer controls](https://github.com/Dhafen/flash-local/actions/runs/39220358126)
also pass on Mac and Linux, covering 256 synthetic file-read waves and 3,584
byte-exact expert jobs across 1/2/4/8 readers, exactly-once execution, persistent
starts, injected errors, reuse after failure and released owners.

Next: expose a separate source-bound phone parity gate and paired at-least-128
token benchmark, recording actual thread configuration, I/O, GPU time, memory
and thermal state. Keep the released engine as the reference. A default change
requires a repeatable physical-phone improvement with unchanged numerics;
neither host functional success nor fewer thread launches proves higher tok/s.
