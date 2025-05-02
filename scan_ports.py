#!/usr/bin/env python3
import socket

# Your interfaces from `ip -4 addr show`
addresses = [
    # "127.0.0.1",        # loopback (always available for single‐node, multi‐process)
    "10.31.144.218",    # your em2 (Ethernet) address
    "10.31.180.218",    # your ib0 (InfiniBand) address
]


# Choose a range to test (here: 29500–29509)
ports = range(29500, 29510)

for addr in addresses:
    for port in ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((addr, port))
            print(f"✔ {addr}:{port} is free")
        except OSError:
            # Port in use or not allowed
            pass
        finally:
            s.close()

# (trdy) bash-4.4$ ip -4 addr show
# 1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
#     inet 127.0.0.1/8 scope host lo
#        valid_lft forever preferred_lft forever
# 3: em2: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP group default qlen 1000
#     altname enp101s0f1np1
#     inet 10.31.144.218/22 brd 10.31.147.255 scope global noprefixroute em2
#        valid_lft forever preferred_lft forever
# 4: ib0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 2044 qdisc mq state UP group default qlen 256
#     inet 10.31.180.218/20 brd 10.31.191.255 scope global noprefixroute ib0
#        valid_lft forever preferred_lft forever
# (trdy) bash-4.4$ 
# (trdy) bash-4.4$ ss -tuln
# Netid       State        Recv-Q        Send-Q               Local Address:Port                Peer Address:Port       Process       
# udp         UNCONN       0             0                          0.0.0.0:5353                     0.0.0.0:*                        
# udp         UNCONN       0             0                          0.0.0.0:36561                    0.0.0.0:*                        
# udp         UNCONN       0             0                          0.0.0.0:40225                    0.0.0.0:*                        
# udp         UNCONN       0             0                          0.0.0.0:51202                    0.0.0.0:*                        
# udp         UNCONN       0             0                          0.0.0.0:111                      0.0.0.0:*                        
# udp         UNCONN       0             0                        127.0.0.1:323                      0.0.0.0:*                        
# udp         UNCONN       0             0                        127.0.0.1:890                      0.0.0.0:*                        
# tcp         LISTEN       0             2048                       0.0.0.0:6818                     0.0.0.0:*                        
# tcp         LISTEN       0             128                        0.0.0.0:43661                    0.0.0.0:*                        
# tcp         LISTEN       0             128                        0.0.0.0:111                      0.0.0.0:*                        
# tcp         LISTEN       0             64                         0.0.0.0:34769                    0.0.0.0:*                        
# tcp         LISTEN       0             6                        127.0.0.1:5555                     0.0.0.0:*                        
# tcp         LISTEN       0             128                        0.0.0.0:22                       0.0.0.0:*                        
# tcp         LISTEN       0             128                      127.0.0.1:631                      0.0.0.0:*                        
# tcp         LISTEN       0             2048                       0.0.0.0:9400                     0.0.0.0:*                        
# tcp         LISTEN       0             127                        0.0.0.0:988                      0.0.0.0:*  