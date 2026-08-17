#!/bin/bash
# Samples CPU die temp (k10temp/Tctl) once a second while a benchmark runs.
# Usage: ./temp_sample.sh /path/to/output.log &   (then kill %1 when the benchmark ends)
while true; do
    ts=$(date +%s)
    temp_millideg=$(cat /sys/class/hwmon/hwmon1/temp1_input)
    echo "$ts $((temp_millideg / 1000))"
    sleep 1
done
