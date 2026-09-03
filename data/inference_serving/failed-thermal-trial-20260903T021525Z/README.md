# Failed thermal-identification trial

This directory preserves the real NVIDIA L4 measurements from the first
thermal-only protocol attempted on 2026-09-02 local time. The independent
safety watchdog stopped the run at 77 degrees C during the first nominal
40 W training plateau. The applied power-limit readback was 40 W, but the
loaded GPU averaged about 47.2 W and remained at its 210 MHz realized-clock
floor. Temperature rose from 49 to 77 degrees C in approximately 377 seconds.

The trial is intentionally marked failed and is not part of the prespecified
training or validation bundle. It is retained as protocol-design evidence and
must not be presented as a completed model-identification experiment.
