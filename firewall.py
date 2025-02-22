import socket
import os
import logging
import threading
from scapy.all import sniff, IP, TCP, UDP

# Configuration
rules = {"192.168.1.100", "10.0.0.200"}  # IPs to block
blocked = {"example.com", "test.net"}     # Domains to block
log_path = "events.log"

# Logging setup
logging.basicConfig(filename=log_path, level=logging.INFO,
               https://www.google.com/search?q=grok+3&ie=UTF-8&oe=UTF-8&hl=en-ph&client=safari#scso=_BVy2Z4uXJsCovr0P4tq76Aw_111:496     format="%(asctime)s - %(levelname)s - %(message)s")

def log_action(entry):
    """Log an action and print it to the console."""
    logging.info(entry)
    print(f"[System] {entry}")

def check_rule(item):
    """Check if an IP is in the rules set."""
    return item in rules

def check_data(data):
    """Check if any blocked domain is in the data."""
    return any(item in data for item in blocked)

def resolve(item):
    """Resolve a domain name to an IP address."""
    try:
        return socket.gethostbyname(item)
    except socket.gaierror:
        return None

def analyze(packet):
    """Analyze a network packet and enforce rules."""
    if IP in packet:
        src = packet[IP].src
        dest = packet[IP].dst

        # Block traffic to/from restricted IPs
        if check_rule(src) or check_rule(dest):
            log_action(f"Blocked {src} -> {dest}")
            return

        # Check payload for blocked domains
        if TCP in packet or UDP in packet:
            content = bytes(packet[TCP].payload).decode(errors="ignore")
            for item in blocked:
                if item in content:
                    log_action(f"Prevented access to {item} from {src}")
                    return

def restrict(item):
    """Block an IP address using system commands."""
    try:
        if os.name == "nt":
            os.system(f"netsh advfirewall firewall add rule name='Restricted' dir=in action=block remoteip={item}")
        else:
            os.system(f"iptables -A INPUT -s {item} -j DROP")
        log_action(f"Restricted {item}")
    except Exception as e:
        log_action(f"Failed to restrict {item}: {e}")

def monitor():
    """Start packet sniffing."""
    log_action("System initialized.")
    sniff(filter="ip", prn=analyze, store=0)

if __name__ == "__main__":
    # Run the monitor in a separate thread
    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()
    log_action("Monitoring started in a separate thread.")
