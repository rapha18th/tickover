---
name: tickover
description: Run a listening routine for machines with no sensors. Enroll healthy sound, schedule weekly voice-note check-ins, check recordings for drift, report the fleet.
---

# Tickover

Tickover gives Hermes an ear for machines that have no sensors and no API: generators, borehole
and water pumps, compressors, fans, older vehicles, workshop machines.

## Setting up a machine
1. `tickover_watch(machine, kind, location, schedule)` saves the profile and schedules the weekly
   check-in in this chat (default every Monday 9am).
2. Ask for healthy recordings: 3 or more of about 10 s, or one of 30 s or more, from the spot the
   owner will use every week. Call `tickover_enroll(machine, paths)`.

## When a recording arrives
- Voice notes and audio files are routed to you with their path when a check-in is expected.
  Call `tickover_check(machine, path)`.
- Put the `show_user` line (`MEDIA:<path>`) on its own line so the card reaches the chat.
- Reply in two or three plain sentences: the verdict, the trend, what to do this week.
- If the verdict is healthy and the owner confirms all is well, call again with
  `add_to_baseline=true` so the baseline keeps up with the seasons.

## Reading a result
- `drift_sigma` under 2: normal. 2 to 3: watch. 3 and above: the sound changed.
- `trend.state`: steady, worsening (rising for weeks), watch or drift. `days_to_drift` is a
  forecast from the recent check-ins; say how many check-ins it rests on.
- `trend.escalate`: two watch or drift readings in the last three, or one drift. Then tell the owner
  plainly to have it inspected and offer a short checklist that fits the kind of machine and the
  band that moved:
  - under 0.6 kHz: mounts, imbalance, loose parts, combustion
  - 0.6 to 3.9 kHz: belts, gears, valves, impeller, tappets
  - 3.9 to 8 kHz: bearings, cavitation, rubbing, whine
- Tickover detects change against the machine's own normal. It never names the fault. Say the
  checklist is a guess.

## Reports
`tickover_fleet()` returns every machine with state, trend and overdue check-ins, plus one card.
Use it when the owner asks how things are, and on Mondays after the check-ins.

## Chat audio
With speech-to-text on, a machine recording often transcribes to nothing. Tickover keeps the file
path and hands it to you anyway. Ignore any transcript of machine audio.
