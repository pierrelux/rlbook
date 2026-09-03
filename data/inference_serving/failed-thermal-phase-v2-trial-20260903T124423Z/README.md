# Failed thermal phase trial (v2)

This bundle preserves the second phase-confirmation trial as diagnostic evidence. It is not an accepted calibration dataset and must not be used by the textbook artifact builder.

The run completed the first 46 W, 75 s decode pulse, then failed closed before the second pulse. The first pulse began at 57 C and peaked at 69 C. Before the second pulse, the 120 s cooldown window satisfied the registered temperature and idle-power criteria, but the one-second memory relock raised the measured temperature from 57 C to 58 C. The profiler compared that post-relock value against the original 56 C pre-relock conditioning reference and aborted. The two actual pulse-start candidates, 57 C and 58 C, remained inside the intended one-degree band and below the 58 C ceiling.

The VM and its disk were deleted after the partial bundle was copied. The follow-up protocol removes that redundant pre-relock/post-relock comparison while retaining the hard 58 C ceiling and the across-pulse one-degree start-temperature constraint.
