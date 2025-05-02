#!/usr/bin/env bash
echo "=== /proc/net/if_inet6 ==="
cat /proc/net/if_inet6 2>/dev/null || echo "(not present)"
echo
echo "=== sysctl disable_ipv6 ==="
sysctl net.ipv6.conf.all.disable_ipv6
echo
echo "=== lsmod ipv6 ==="
lsmod | grep ipv6 || echo "(module not loaded)"
echo
echo "=== ip -6 route ==="
ip -6 route 2>/dev/null || echo "(no IPv6 routes)"
