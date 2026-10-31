# SPEC task54 — `uptake-perturb` graded variants (Amendment 13)
Add `--levels slight,strong` to `write` (default stays rude,polite): SLIGHT = one small hardening touch on the original
(prompt constant), STRONG = the existing RUDE prompt. `vote --variants` must accept any variant keys present and build pairs
(orig,<a>), (orig,<b>), (<a>,<b>). `report` prints, when the keys are slight/strong: targets with slight > strong (x/20), mean
margin orig-vs-slight and orig-vs-strong, and the Amendment-13 verdict line. tests: write with levels stores both keys;
vote pairs follow the keys; report verdict on a fixture. `pytest -q` green.
