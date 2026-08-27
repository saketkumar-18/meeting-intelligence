#!/usr/bin/env bash
# Resumable download: keeps retrying with -C - until the file is complete.
set -u
url="$1"
out="$2"
for i in $(seq 1 40); do
  curl -sL --connect-timeout 20 --max-time 240 -C - -o "$out" "$url"
  code=$?
  if [ $code -eq 0 ] || [ $code -eq 33 ]; then
    # 0 = done; 33 = range already satisfied (file already complete)
    echo "COMPLETE $out ($(stat -c %s "$out" 2>/dev/null || stat -f %z "$out") bytes)"
    exit 0
  fi
  echo "attempt $i curl=$code, resuming..."
  sleep 2
done
echo "FAILED after 40 attempts"
exit 1
