#!/bin/sh
set -eu
host=${1:-gx10-a}
port=${2:-8765}
case "$host" in gx10-a|gx10-b) ;; *) echo 'Use gx10-a or gx10-b'; exit 2;; esac
case "$port" in 8765|8766) ;; *) echo 'Use local port 8765 or 8766'; exit 2;; esac
echo "Dashboard: http://127.0.0.1:$port (keep this terminal open)"
exec ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -L "127.0.0.1:$port:127.0.0.1:8765" -N "$host"
