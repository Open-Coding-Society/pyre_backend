from api.jwt_authorize import token_required
from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
import subprocess
import re

# Blueprint for Poll API
health_api = Blueprint('health_api', __name__, url_prefix='/api')
api = Api(health_api)

class HealthAPI(Resource):
    def get(self):
        try:
            # Run system commands with elevated privileges where needed
            raw_ram = subprocess.check_output(["sudo", "free", "-h"], stderr=subprocess.DEVNULL).decode("utf-8")
            raw_cpu = subprocess.check_output(["sudo", "top", "-bn1"], stderr=subprocess.DEVNULL).decode("utf-8")
            raw_disk = subprocess.check_output(["sudo", "df", "-h"], stderr=subprocess.DEVNULL).decode("utf-8")
            raw_network = subprocess.check_output(["sudo", "ip", "addr"], stderr=subprocess.DEVNULL).decode("utf-8")
            # New: capture htop output
            raw_htop = subprocess.check_output(["sudo", "htop", "-n", "1", "-CUM"], stderr=subprocess.DEVNULL).decode("utf-8")
        except subprocess.CalledProcessError as e:
            return {
                "error": "Permission denied or command failed",
                "output": str(e)  # or e.output.decode() if you want the command output
            }, 500

        # Parse RAM (only total, used, free)
        ram_info = {}
        ram_line = re.search(r'Mem:\s+(\S+)\s+(\S+)\s+(\S+)', raw_ram)
        if ram_line:
            ram_info["total"] = ram_line.group(1)
            ram_info["used"] = ram_line.group(2)
            ram_info["free"] = ram_line.group(3)

        # Parse CPU (just idle, user, system)
        cpu_info = {}
        cpu_line = re.search(
            r'%Cpu\(s\):\s+([\d\.]+)\s+us,\s+([\d\.]+)\s+sy,\s+[\d\.]+\s+ni,\s+([\d\.]+)\s+id,.*',
            raw_cpu
        )
        if cpu_line:
            cpu_info["user"] = cpu_line.group(1)
            cpu_info["system"] = cpu_line.group(2)
            cpu_info["idle"] = cpu_line.group(3)

        # Parse Disk (example: only root device usage)
        disk_info = []
        for line in raw_disk.splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 6 and fields[0].startswith("/dev/"):
                disk_info.append({
                    "filesystem": fields[0],
                    "size": fields[1],
                    "used": fields[2],
                    "avail": fields[3],
                    "use%": fields[4]
                })

        # Parse Network (just interface and inet lines)
        network_info = []
        for block in raw_network.split('\n\n'):
            if block.strip():
                iface_line = block.split('\n')[0].strip()
                inet_lines = [l.strip() for l in block.split('\n') if "inet " in l]
                network_info.append({
                    "interface": iface_line.split(':')[1].strip() if ':' in iface_line else iface_line,
                    "addresses": inet_lines
                })

        # Parse htop
        htop_info = {}
        match_tasks = re.search(r'Tasks:\s*(\d+),\s*(\d+)\s+thr,\s*(\d+)\s+kthr;\s*(\d+)\s+running', raw_htop)
        if match_tasks:
            htop_info["tasks_total"] = match_tasks.group(1)
            htop_info["tasks_thr"] = match_tasks.group(2)
            htop_info["tasks_kthr"] = match_tasks.group(3)
            htop_info["tasks_running"] = match_tasks.group(4)

        match_load = re.search(r'Load average:\s*([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)', raw_htop)
        if match_load:
            htop_info["load_avg_1"] = match_load.group(1)
            htop_info["load_avg_5"] = match_load.group(2)
            htop_info["load_avg_15"] = match_load.group(3)

        match_mem = re.search(r'Mem\[.*([\d\.]+[GMK])\/([\d\.]+[GMK]).*\]\s*Uptime:\s*(\d+)\s*days,\s*([\d]+):([\d]+):([\d]+)', raw_htop)
        if match_mem:
            htop_info["mem_used"] = match_mem.group(1)
            htop_info["mem_total"] = match_mem.group(2)
            htop_info["uptime_days"] = match_mem.group(3)
            htop_info["uptime_hours"] = match_mem.group(4)
            htop_info["uptime_mins"] = match_mem.group(5)
            htop_info["uptime_secs"] = match_mem.group(6)

        match_swp = re.search(r'Swp\[.*(\d+[KMG])\/(\d+[KMG])\]', raw_htop)
        if match_swp:
            htop_info["swap_used"] = match_swp.group(1)
            htop_info["swap_total"] = match_swp.group(2)

        return {
            "ram": ram_info,
            "cpu": cpu_info,
            "disk": disk_info,
            "network": network_info,
            "htop": htop_info
        }, 200

api.add_resource(HealthAPI, "/health")
