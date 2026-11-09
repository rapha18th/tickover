---
name: earshot
description: Listen to machines. Enroll a machine's healthy sound, then check new recordings for drift with the earshot_* tools.
---

# Earshot

Use Earshot when the user sends or mentions a recording of a machine (robot joint, servo,
drone, fan, 3D printer, pump, compressor, generator, car engine, PC or GPU fan) and wants
to know whether it sounds different from normal.

## Flow
1. **No baseline yet:** ask for 3+ healthy recordings of about 10 s, or one of 30 s or more,
   taken from the same spot and distance. Call `earshot_enroll(machine, paths)`.
2. **Check:** call `earshot_check(machine, path)`. Put the returned `show_user` line
   (`MEDIA:<path>`) on its own line in your reply so the card appears in chat.
3. **Machine next to the computer:** `earshot_listen(seconds=10, machine, then="check")`.
4. **History:** `earshot_status(machine)`.

## Reading the result
- `drift_sigma` under 2: normal variation. 2 to 3: watch. 3 and above: the sound changed.
- `most_changed_band` says where. Low bands (under 0.6 kHz) point to imbalance, mounts,
  loose parts or combustion. Mid bands (0.6-3.9 kHz) to gears, belts, valves, tappets.
  High bands (3.9-8 kHz) to bearings, brushes, rubbing, whine and cavitation.
- Earshot detects change against the machine's own normal. It never names a fault.
  Offer an inspection checklist that fits the band and the machine. Say it is a guess.
- Voice notes in chat go through speech-to-text by default. For machine sound, send the
  recording as a file, or set `stt.enabled: false` for this profile.
