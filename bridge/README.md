# Tickover bridge

A spare device with a microphone becomes an ear Hermes can listen to. One Python file, standard library only.

```
GET /clip?seconds=10   records now and returns a 16 kHz mono WAV (3 to 60 s)
GET /latest            the newest clip from the --every timer, with X-Tickover-Age
GET /health            JSON: name, backend, last level in dBFS, last clip age, uptime
GET /                  a short page that lists these URLs
```

On start it prints its URLs and a line to paste to Hermes. It needs `--key SECRET`, or `--open` to let anyone on the network listen. Both setup scripts make a key for you.

## Old Android phone

1. Install **Termux**, **Termux:API** and **Termux:Boot** from F-Droid. Open Termux:Boot once.
2. In Termux:

```sh
curl -LO https://raw.githubusercontent.com/rapha18th/tickover/main/bridge/tickover_bridge.py
curl -LO https://raw.githubusercontent.com/rapha18th/tickover/main/bridge/termux_setup.sh
sh termux_setup.sh --name "borehole pump"
```

The setup records a 3 s test. Allow the microphone when Android asks. It makes a key, starts the bridge, restarts it if it ever exits, and writes `~/.termux/boot/tickover-bridge` so it starts again after a reboot.

Keep the phone on the charger and on Wi-Fi. Turn off battery optimisation for Termux and Termux:API. The script holds a wake lock so recording continues with the screen off. If a check says the device gave silence, Android has paused the microphone: keep Termux open on screen and check the permission.

## Raspberry Pi with a USB microphone

```sh
curl -LO https://raw.githubusercontent.com/rapha18th/tickover/main/bridge/tickover_bridge.py
curl -LO https://raw.githubusercontent.com/rapha18th/tickover/main/bridge/install_pi.sh
curl -LO https://raw.githubusercontent.com/rapha18th/tickover/main/bridge/tickover-bridge.service
sudo sh install_pi.sh --name "borehole pump" --every 600
```

It installs `alsa-utils`, finds the USB microphone, makes a key, copies the bridge to `/opt/tickover-bridge`, and enables a systemd service that starts on boot and restarts on failure. Then it records a 3 s test clip and prints the URLs. Run it again to change a setting. Settings you leave out keep their saved values.

With `--every`, give Hermes the `/latest` URL. Hermes then also notices when the device stops recording. Timed clips are 10 s or longer, because checks compare 10-second clips.

If the Pi has more than one microphone, list them with `arecord -l` and add `--device plughw:1,0`. On Pi OS Desktop, if arecord says the device is busy, use Pi OS Lite or close apps that use the microphone.

## Any laptop or mini PC

```sh
curl -LO https://raw.githubusercontent.com/rapha18th/tickover/main/bridge/tickover_bridge.py
python3 tickover_bridge.py --name "workshop compressor" --key CHOOSE_ONE
```

On Windows, use `py tickover_bridge.py ...`. Windows 10 and later ship curl.

It finds a way to record: `arecord` on Linux, the `sounddevice` package (`pip install sounddevice numpy`), or ffmpeg (ALSA or PulseAudio on Linux, AVFoundation on macOS, DirectShow on Windows). Force one with `--backend`. Pick the input with `--device`.

Set the laptop to never sleep on power. Keep the lid open, or set closing the lid to do nothing.

No microphone yet? `--replay FOLDER` serves recordings from a folder in a loop. Use it to try the setup end to end.

## Can Hermes reach it?

Open the `/health` URL from the computer Hermes runs on. If it loads, you are done.

- Hermes on a server or VPS: install Tailscale on the server and on this device, and use the device's 100.x address.
- Windows: allow Python on Private networks when Defender asks, and set the Wi-Fi to Private.
- macOS: click Allow when asked to accept incoming connections.
- Linux with ufw: `sudo ufw allow 8765/tcp`.
- Guest Wi-Fi often blocks devices from seeing each other. Use the main network.

## What to tell Hermes

```
Watch the borehole pump from http://192.168.1.20:8765/clip?key=CHOOSE_ONE every morning at 7.
```

Hermes records a fresh clip on schedule and speaks only when the sound drifts. The same URL works for verify: "Switch the pump on and listen through http://192.168.1.20:8765/clip?key=CHOOSE_ONE."

With `--every`, `/latest` always holds a recent clip. If the device stops recording, the check returns `stale` and Hermes tells you. If the device does not answer, Hermes says the listening device at that URL is not answering. Hermes hides the key when it repeats the URL.

## Placement

- Same spot, same distance, every time. Tickover compares a machine against its own normal, so a moved microphone reads as a change.
- Close to the machine. Half a metre beats three metres.
- Out of the wind, off vibrating panels, away from other machines where you can.
- Enroll from this device, at this spot, before you watch.

## Security

- Every request needs `?key=...` unless you start it with `--open`. A wrong key gets 401.
- The bridge serves only audio it records now, or the files you pass with `--replay`. No URL reads a file on the device.
- It answers GET and HEAD only, records one clip at a time, queues at most three, and never runs a shell.
- Keep it on your local network or a Tailscale network. Do not forward the port to the internet.
