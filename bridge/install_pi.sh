#!/bin/sh
# Turn a Raspberry Pi (or any Debian machine) with a USB microphone into a Tickover ear.
#
#   sudo sh install_pi.sh [--key SECRET] [--every SECONDS] [--seconds N] [--name "borehole pump"]
#                         [--port 8765] [--device plughw:1,0]
#
# Safe to run again. Settings you leave out keep their saved values, the key included.
# It installs alsa-utils, finds the USB microphone, copies the bridge to /opt/tickover-bridge,
# writes /etc/default/tickover-bridge, enables a systemd service that restarts on failure,
# and records a 3 s test clip. With no --key it makes one. Checks need --seconds 10 or more.
set -eu

KEY="" EVERY="" SECONDS_="" NAME="" PORT="" DEVICE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --key) KEY="$2"; shift 2 ;;
    --every) EVERY="$2"; shift 2 ;;
    --seconds) SECONDS_="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [ "$(id -u)" -ne 0 ]; then
  echo "Run this as root." >&2
  exit 1
fi

# A re-run keeps what was saved. Read the old values as data. Nothing in the file is executed.
CONF=/etc/default/tickover-bridge
if [ -f "$CONF" ]; then
  OLD_ARGS="$(sed -n 's/^TICKOVER_BRIDGE_ARGS=//p' "$CONF")"
  old() { printf '%s\n' "$OLD_ARGS" | tr ' ' '\n' | sed -n "/^$1\$/{n;p;}" | head -n1; }
  [ -n "$KEY" ] || KEY="$(sed -n 's/^TICKOVER_BRIDGE_KEY=//p' "$CONF")"
  [ -n "$NAME" ] || NAME="$(sed -n 's/^TICKOVER_BRIDGE_NAME="\(.*\)"$/\1/p' "$CONF")"
  [ -n "$EVERY" ] || EVERY="$(old --every)"
  [ -n "$SECONDS_" ] || SECONDS_="$(old --seconds)"
  [ -n "$PORT" ] || PORT="$(old --port)"
  [ -n "$DEVICE" ] || DEVICE="$(old --device)"
fi
[ -n "$PORT" ] || PORT="8765"

case "$KEY" in *[!A-Za-z0-9_-]*) echo "--key may use letters, digits, - and _ only" >&2; exit 2 ;; esac
case "$NAME" in *[!A-Za-z0-9_.\ -]*) echo "--name may use letters, digits, spaces, dots, - and _ only" >&2; exit 2 ;; esac
for v in "$EVERY" "$SECONDS_" "$PORT"; do
  case "$v" in *[!0-9.]*) echo "--every, --seconds and --port take numbers" >&2; exit 2 ;; esac
done
case "$DEVICE" in *[!A-Za-z0-9:,_.=-]*) echo "--device looks wrong. Example: plughw:1,0" >&2; exit 2 ;; esac

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/tickover_bridge.py"
UNIT="$HERE/tickover-bridge.service"
[ -f "$SRC" ] || { echo "tickover_bridge.py must sit next to this script" >&2; exit 1; }

command -v python3 >/dev/null 2>&1 || { apt-get update -q && apt-get install -y -q python3; }
command -v arecord >/dev/null 2>&1 || { apt-get update -q && apt-get install -y -q alsa-utils; }

# No key given or saved: make one. token_urlsafe uses letters, digits, - and _ only.
[ -n "$KEY" ] || KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(18))')"

# No device given: the first card that can capture, through plughw so ALSA resamples to 16 kHz.
if [ -z "$DEVICE" ]; then
  DEVICE="$(arecord -l 2>/dev/null | sed -n 's/^card [0-9]*: \([A-Za-z0-9_]*\) .*/plughw:CARD=\1,DEV=0/p' | head -n1)"
  if [ -z "$DEVICE" ]; then
    echo "No microphone found. Plug in a USB microphone and run again." >&2
    exit 1
  fi
fi

id tickover >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin --groups audio tickover
usermod -a -G audio tickover

install -d -m 755 /opt/tickover-bridge
install -m 755 "$SRC" /opt/tickover-bridge/tickover_bridge.py

# Every value was checked above. systemd splits TICKOVER_BRIDGE_ARGS into argv. No shell runs it.
ARGS="--port $PORT --backend arecord --device $DEVICE"
[ -n "$EVERY" ] && ARGS="$ARGS --every $EVERY"
[ -n "$SECONDS_" ] && ARGS="$ARGS --seconds $SECONDS_"
TMPF="$(mktemp)"
trap 'rm -f "$TMPF"' EXIT
umask 077
{
  echo "TICKOVER_BRIDGE_ARGS=$ARGS"
  if [ -n "$NAME" ]; then echo "TICKOVER_BRIDGE_NAME=\"$NAME\""; fi
  echo "TICKOVER_BRIDGE_KEY=$KEY"
} > "$TMPF"
install -m 640 -o root -g root "$TMPF" "$CONF"

if [ -f "$UNIT" ]; then
  install -m 644 "$UNIT" /etc/systemd/system/tickover-bridge.service
fi
systemctl daemon-reload
systemctl enable tickover-bridge.service >/dev/null
systemctl restart tickover-bridge.service

sleep 2
if ! systemctl is-active --quiet tickover-bridge.service; then
  echo "The bridge did not start. See: journalctl -u tickover-bridge -n 30" >&2
  exit 1
fi

# Proof the microphone works: a 3 s clip through the bridge itself.
if ! python3 - "$PORT" "$KEY" <<'EOF'
import sys, urllib.request
port, key = sys.argv[1], sys.argv[2]
r = urllib.request.urlopen(f"http://127.0.0.1:{port}/clip?seconds=3&key={key}", timeout=90)
print("Microphone OK.", len(r.read()), "bytes recorded.")
EOF
then
  echo "The bridge runs but could not record. List microphones with: arecord -l" >&2
  echo "Then run again with --device plughw:CARD,0. Details: journalctl -u tickover-bridge -n 30" >&2
  exit 1
fi

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
[ -n "$IP" ] || IP="$(hostname).local"
Q="?key=$KEY"
echo
echo "Tickover bridge is running and starts on boot. Microphone: $DEVICE"
echo "  Clip now:  http://$IP:$PORT/clip$Q"
[ -n "$EVERY" ] && echo "  Newest:    http://$IP:$PORT/latest$Q"
echo "  Health:    http://$IP:$PORT/health$Q"
echo
echo "Tell Hermes:"
if [ -n "$EVERY" ]; then
  echo "  Watch the ${NAME:-machine} from http://$IP:$PORT/latest$Q every morning at 7."
else
  echo "  Watch the ${NAME:-machine} from http://$IP:$PORT/clip$Q every morning at 7."
fi
echo
echo "Open the Health URL from the computer Hermes runs on. If it loads, Hermes can reach this device."
