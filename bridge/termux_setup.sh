#!/data/data/com.termux/files/usr/bin/sh
# Turn an old Android phone into a Tickover ear.
#
# 1. Install Termux, Termux:API and Termux:Boot from F-Droid (the Play Store builds are outdated).
#    Open Termux:Boot once, so Android lets it start the bridge after a reboot.
# 2. In Termux:
#      curl -LO https://raw.githubusercontent.com/rapha18th/tickover/main/bridge/tickover_bridge.py
#      curl -LO https://raw.githubusercontent.com/rapha18th/tickover/main/bridge/termux_setup.sh
#      sh termux_setup.sh [--key SECRET] [--every 600] [--name "borehole pump"]
#
# The setup records a 3 s test. Allow the microphone when Android asks. With no --key it makes
# one. It writes ~/.termux/boot/tickover-bridge, so the bridge starts again after a reboot, and
# restarts it if it ever exits.
#
# Keep the phone plugged in and on Wi-Fi. In Android settings, turn off battery optimisation
# for Termux and Termux:API, or Android will pause the bridge. Tape or mount the phone in one
# spot, close to the machine, microphone facing it.
set -eu

pkg install -y python termux-api ffmpeg

HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$HERE/tickover_bridge.py" ] || { echo "tickover_bridge.py must sit next to this script" >&2; exit 1; }

# No --key given: make one.
case " $* " in
  *" --key "*|*" --open "*) ;;
  *) set -- "$@" --key "$(python -c 'import secrets; print(secrets.token_urlsafe(18))')" ;;
esac

# Record a test clip while someone is watching, so Android can show the permission dialog.
TEST="${TMPDIR:-$HOME}/tickover-test.m4a"
rm -f "$TEST"
echo "Recording 3 s to test the microphone. Allow the permission if Android asks."
termux-microphone-record -q >/dev/null 2>&1 || true
termux-microphone-record -f "$TEST" -l 3 >/dev/null 2>&1 || true
sleep 5
termux-microphone-record -q >/dev/null 2>&1 || true
if [ ! -s "$TEST" ]; then
  echo "No audio. Allow the microphone for Termux:API in Android settings and run again." >&2
  exit 1
fi
rm -f "$TEST"
echo "Microphone OK."

# The start line, with every argument quoted, for the boot script.
QARGS="$(python -c 'import shlex, sys; print(" ".join(shlex.quote(a) for a in sys.argv[1:]))'   "$HERE/tickover_bridge.py" --backend termux "$@")"
BOOT="$HOME/.termux/boot"
mkdir -p "$BOOT"
{
  echo "#!/data/data/com.termux/files/usr/bin/sh"
  echo "termux-wake-lock"
  echo "while true; do python $QARGS; sleep 5; done"
} > "$BOOT/tickover-bridge"
chmod 700 "$BOOT/tickover-bridge"
echo "Wrote $BOOT/tickover-bridge. The bridge starts again after a reboot."

# Keep the CPU awake while the screen is off.
termux-wake-lock

echo "Starting the bridge. Leave Termux open in the background. Ctrl+C twice stops it."
echo "It prints a 'hermes config set' command. Run that once on the computer Hermes runs on to allow this phone,"
echo "then give Hermes the Watch line it prints."
while true; do
  python "$HERE/tickover_bridge.py" --backend termux "$@" || true
  echo "The bridge stopped. Starting it again in 5 s."
  sleep 5
done
