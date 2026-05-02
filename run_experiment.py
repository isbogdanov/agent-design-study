
import os
import yaml
import shutil
import argparse
import subprocess
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from glob import glob
import re
import sys
import threading
import atexit

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ExperimentRunner")

# ─────────────────────────────────────────────────────────────────────────────
# Optional rich import — used only when --progress flag is active
# ─────────────────────────────────────────────────────────────────────────────
try:
    from rich.live import Live as _RichLive
    from rich.table import Table as _RichTable
    from rich.console import Console as _RichConsole
    from rich import box as _rich_box
    _RICH = True
except ImportError:
    _RICH = False


class ProgressMonitor:
    """
    Monitors running Docker evaluation instances by incrementally reading their
    docker.log files.  Activated ONLY when --progress CLI flag is passed.

    Parses three signals from docker.log:
      MAIN_LOOP| Episode=1 | Step=S  →  current step within the active run
      --- EVALUATION RUN N ---       →  run number boundary (count + 1 = current run)
      Total Cumulative Reward: X     →  reward of the just-completed episode

    Uses an incremental file-cursor (like tail -f) so no lines are missed
    even when a single step produces tens of KB of LLM output.
    """

    _MAINLOOP_PAT = re.compile(
        r'MAIN_LOOP\| Episode=\d+ \| Step=(\d+).*?\| Reward=([\-\d\.]+) \| TotalReward=([\-\d\.]+)'
    )
    _EVALRUN_PAT  = re.compile(r'--- EVALUATION RUN (\d+) ---')
    _REWARD_PAT   = re.compile(r'Total Cumulative Reward:\s+([\-\d\.]+)')

    def __init__(self, workspaces_dir, num_instances, num_eval_runs, total_steps=30):
        self.workspaces_dir = workspaces_dir
        self.num_instances  = num_instances
        self.num_eval_runs  = num_eval_runs
        self.total_steps    = total_steps
        self._global_start  = time.time()
        self._stop          = threading.Event()
        self._thread        = None
        self._live          = None

        # Per-instance mutable state
        self._state = {
            i: {
                'run': 1, 'step': 0,
                'step_reward': None, 'total_reward': None,
                'episode_completed': False,  # True once 'Total Cumulative Reward:' seen
                'rewards': [], 'start': time.time()
            }
            for i in range(1, num_instances + 1)
        }
        # Incremental read cursor: bytes already consumed per instance docker.log
        self._pos = {i: 0 for i in range(1, num_instances + 1)}

    # ── public API ────────────────────────────────────────────────────────────

    def start(self):
        # Log before starting the live display so the message isn't interleaved
        # with rich's rendering and causes a doubled-title artifact.
        logger.info("Progress monitor started (polling every 2 s).")
        if _RICH:
            self._live = _RichLive(
                self._make_table(),
                console=_RichConsole(),
                refresh_per_second=1,
                transient=False,
            )
            self._live.start()
            # Ensure the live display is closed cleanly even on sys.exit(1)
            # (the SIGINT handler calls sys.exit directly, bypassing KeyboardInterrupt)
            atexit.register(self._close_live)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _close_live(self):
        """Idempotent: close the rich.Live display. Called via atexit on sys.exit.

        Stops the background thread first to eliminate the race between the
        thread's _live.update() call and our _live.stop() call here.
        """
        # Signal the loop thread to exit and wait briefly
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        # Final poll: capture any log lines written after the last background poll
        # (the last episode's MAIN_LOOP may land after the thread's final iteration)
        self._poll()
        if self._live:
            self._live.update(self._make_table())  # final refresh
        self._close_live()

    # ── internals ─────────────────────────────────────────────────────────────

    def _docker_log_path(self, instance_id):
        return os.path.join(
            self.workspaces_dir, f"instance_{instance_id}", "docker.log"
        )

    def _poll(self):
        """Read only newly appended bytes from each docker.log and update state."""
        for i in range(1, self.num_instances + 1):
            path = self._docker_log_path(i)
            if not os.path.exists(path):
                continue
            try:
                with open(path, 'r', errors='ignore') as f:
                    f.seek(self._pos[i])
                    new_text = f.read()
                    self._pos[i] = f.tell()
                if not new_text:
                    continue

                # Run-number boundary: each separator marks start of run N
                run_matches = self._EVALRUN_PAT.findall(new_text)
                if run_matches:
                    # Archive running total only if the episode-end summary wasn't seen
                    # (avoids double-counting alongside the _REWARD_PAT branch below)
                    if not self._state[i]['episode_completed'] and self._state[i]['total_reward'] is not None:
                        self._state[i]['rewards'].append(self._state[i]['total_reward'])
                    self._state[i]['run'] = int(run_matches[-1])
                    self._state[i]['step'] = 0
                    self._state[i]['step_reward'] = None
                    self._state[i]['total_reward'] = None   # blank only on new-run start
                    self._state[i]['episode_completed'] = False

                # Current step + per-step reward + running TotalReward
                step_matches = self._MAINLOOP_PAT.findall(new_text)
                if step_matches:
                    last_step, last_step_rew, last_total = step_matches[-1]
                    self._state[i]['step']              = int(last_step)
                    self._state[i]['step_reward']       = float(last_step_rew)
                    self._state[i]['total_reward']      = float(last_total)
                    self._state[i]['episode_completed'] = False

                # Episode-end summary line — archive reward, keep total_reward visible
                for r in self._REWARD_PAT.findall(new_text):
                    self._state[i]['rewards'].append(float(r))
                    self._state[i]['step_reward']       = None   # per-step no longer meaningful
                    self._state[i]['episode_completed'] = True
                    # total_reward intentionally NOT cleared: keeps the final episode
                    # total visible in the dashboard until the next run's MAIN_LOOP fires

            except Exception:
                pass

    def _elapsed(self, since):
        secs = int(time.time() - since)
        m, s = divmod(secs, 60)
        h, m = divmod(m, 60)
        return f"{h}h {m:02d}m {s:02d}s" if h else f"{m:02d}m {s:02d}s"

    def _bar(self, current, total, width=20):
        if total <= 0:
            return '[' + '░' * width + ']'
        filled = min(width, int(width * current / total))
        return '[' + '█' * filled + '░' * (width - filled) + ']'

    def _make_table(self):
        """Build a rich Table representing current per-instance state."""
        t = _RichTable(
            title=f"Experiment Progress  ·  {self._elapsed(self._global_start)} elapsed",
            box=_rich_box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            border_style="bright_black",
        )
        t.add_column("Instance",     style="bold white", justify="center", min_width=10)
        t.add_column("Run",                              justify="center", min_width=8)
        t.add_column("Step Progress",                    justify="left",   min_width=32)
        t.add_column("Elapsed",                          justify="right",  min_width=10)
        t.add_column("Step Rew / Total",                 justify="right",  min_width=16)

        for i in range(1, self.num_instances + 1):
            s = self._state[i]
            run_str  = f"{s['run']}/{self.num_eval_runs}"
            bar_str  = self._bar(s['step'], self.total_steps)
            done     = s['step']
            active   = min(done + 1, self.total_steps)
            if done == 0:
                step_str = f"{bar_str}  [dim]—[/dim]  [bold cyan]▶ {active}/{self.total_steps}[/bold cyan]"
            elif done == self.total_steps:
                step_str = f"{bar_str}  [green]✓ {done}/{self.total_steps}[/green]"
            else:
                step_str = f"{bar_str}  [dim]{done}✓[/dim]  [bold cyan]▶ {active}/{self.total_steps}[/bold cyan]"
            elapsed  = self._elapsed(s['start'])

            sr = s['step_reward']
            tr = s['total_reward']
            if tr is not None:
                color = ("green"       if tr > -50  else
                         "yellow"      if tr > -100 else
                         "dark_orange" if tr > -150 else
                         "red")
                sr_str = f"{sr:.1f}" if sr is not None else "—"
                reward_str = f"[dim]{sr_str}[/dim] / [{color}]{tr:.1f}[/{color}]"
            else:
                reward_str = "[dim]— / —[/dim]"

            t.add_row(f"[bold]#{i}[/bold]", run_str, step_str, elapsed, reward_str)

        return t

    def _log_snapshot(self):
        """Plain-text fallback: emit a progress snapshot via logger.info."""
        lines = [f"Progress ({self._elapsed(self._global_start)} elapsed):"]
        for i in range(1, self.num_instances + 1):
            s = self._state[i]
            done  = s['step']
            active = min(done + 1, self.total_steps)
            if done == 0:
                step_lbl = f"▶ {active}/{self.total_steps}"
            elif done == self.total_steps:
                step_lbl = f"✓ {done}/{self.total_steps}"
            else:
                step_lbl = f"{done}✓ ▶ {active}/{self.total_steps}"
            sr = s['step_reward']
            tr = s['total_reward']
            sr_str = f"{sr:.1f}" if sr is not None else "—"
            tr_str = f"{tr:.1f}" if tr is not None else "—"
            lines.append(
                f"  instance_{i} | Run {s['run']}/{self.num_eval_runs} | "
                f"{bar} {step_lbl} | "
                f"Elapsed {self._elapsed(s['start'])} | Step {sr_str} / Total {tr_str}"
            )
        logger.info("\n".join(lines))

    def _loop(self):
        _snapshot_interval = 10  # seconds between plain-text snapshots (no-rich mode)
        _last_snapshot = time.time()
        while not self._stop.is_set():
            self._poll()
            if self._live:
                self._live.update(self._make_table())
            else:
                now = time.time()
                if now - _last_snapshot >= _snapshot_interval:
                    self._log_snapshot()
                    _last_snapshot = now
            self._stop.wait(2.0)


def _worker_suppress_logging():
    """ProcessPoolExecutor initializer: remove console handlers from the root logger.

    Worker processes inherit the parent's stdout file descriptor.  When --progress
    is active, rich.Live owns that terminal and tracks the cursor.  Any writes from
    worker-process logger.info() calls land *outside* rich's knowledge and shift the
    cursor, causing duplicate table renders.  Workers write all real output to
    docker.log via subprocess stdout redirect, so removing the StreamHandler here
    loses nothing visible to the user.

    Must be a module-level function (not a lambda/closure) to be picklable by the
    'spawn' multiprocessing start method used on macOS.
    """
    root = logging.getLogger()
    root.handlers = [h for h in root.handlers
                     if not isinstance(h, logging.StreamHandler)]


# Module-level reference to the active ProgressMonitor.
# Set in run_regular_experiment() when --progress is active so the signal
# handler can stop it (restoring the terminal) before logging any messages.
_active_monitor = None


# Function to load .env file manually to avoid external dependencies
def load_env_file(filepath=".env"):
    """Loads environment variables from a .env file."""
    if not os.path.exists(filepath):
        return
    
    logger.info(f"Loading environment variables from {filepath}")
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            # Simple handling for KEY=VALUE or KEY="VALUE"
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                
                # Remove surrounding quotes if present
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                
                os.environ[key] = value

def parse_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_experiment_dirs(experiment_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"experiments/{experiment_name}_{timestamp}"
    
    workspaces_dir = os.path.join(base_dir, "workspaces")
    aggregated_dir = os.path.join(base_dir, "aggregated_logs")
    
    os.makedirs(workspaces_dir, exist_ok=True)
    os.makedirs(aggregated_dir, exist_ok=True)
    
    return base_dir, workspaces_dir, aggregated_dir, timestamp

def prepare_instance(instance_id, workspaces_dir, definitions_source_dir, agent_folder="agent_base"):
    """
    Creates the instance directory structure and copies initial knowledge (definitions).
    """
    instance_dir = os.path.join(workspaces_dir, f"instance_{instance_id}")
    definitions_dest = os.path.join(instance_dir, "definitions")
    logs_dest = os.path.join(instance_dir, "logs")
    
    os.makedirs(instance_dir, exist_ok=True)
    os.makedirs(logs_dest, exist_ok=True)
    
    # helper for recursive copy definitions
    if os.path.exists(definitions_dest):
        shutil.rmtree(definitions_dest)
    shutil.copytree(definitions_source_dir, definitions_dest)
    
    # helper for recursive copy log config
    # We must preserve logs/config because mounting 'logs' hides the code version
    log_config_src = os.path.abspath(f"{agent_folder}/logs/config")
    log_config_dest = os.path.join(logs_dest, "config")
    
    if os.path.exists(log_config_src):
        if os.path.exists(log_config_dest):
            shutil.rmtree(log_config_dest)
        shutil.copytree(log_config_src, log_config_dest)
    else:
        logger.warning(f"Log config source not found at {log_config_src}")
    
    # helper for connector logs
    # We create a specific directory for connector logs and mount it
    connector_logs_dest = os.path.join(logs_dest, "connector")
    os.makedirs(connector_logs_dest, exist_ok=True)

    return instance_dir, logs_dest, definitions_dest, connector_logs_dest

def construct_docker_command(instance_id, logs_host_path, definitions_host_path, connector_logs_host_path, config, agent_folder="agent_base"):
    """Constructs the Docker run command."""
    agent_config = config['agent_config']
    
    # Base command structure
    cmd = [
        "docker", "run",
        "--rm",  # Remove container after exit
        "--name", f"cyborg_worker_{config.get('experiment_timestamp', 'default')}_{instance_id}",
        # Mount the code (read-only)
        "-v", f"{os.path.abspath(agent_folder)}:/app/agent_base:ro",
        # Mount the logs directory (read-write)
        "-v", f"{os.path.abspath(logs_host_path)}:/app/agent_base/logs",
        # Mount the connector logs directory (read-write)
        "-v", f"{os.path.abspath(connector_logs_host_path)}:/app/agent_base/llm-connector/logs",
        # Mount the definitions directory (read-write, overlay)
        "-v", f"{os.path.abspath(definitions_host_path)}:/app/agent_base/agents/prompts/definitions",
        # Environment variables
        "-e", "PYTHONUNBUFFERED=1",
    ]
    
    # API Key Handling
    # Default mappings
    display_keys = {
        "GOOGLE_API_KEY": os.environ.get('GOOGLE_API_KEY', ''),
        "OPENROUTER_API_KEY": os.environ.get('OPENROUTER_API_KEY', ''),
        "OPENAI_API_KEY": os.environ.get('OPENAI_API_KEY', ''),
        "VERTEX_API_KEY": os.environ.get('VERTEX_API_KEY', '')
    }
    
    # Custom Override
    custom_source = agent_config.get('api_key_env_var')
    if custom_source:
        provider = agent_config.get('provider', '').lower()
        target_var = None
        if provider == 'google': target_var = 'GOOGLE_API_KEY'
        elif provider == 'openai': target_var = 'OPENAI_API_KEY'
        elif provider == 'openrouter': target_var = 'OPENROUTER_API_KEY'
        elif provider == 'vertex': target_var = 'VERTEX_API_KEY'
        
        if target_var:
            val = os.environ.get(custom_source)
            if val:
                display_keys[target_var] = val
            else:
                logger.warning(f"Custom key source '{custom_source}' not found in environment.")

    for k, v in display_keys.items():
        cmd.extend(["-e", f"{k}={v}"])

    cmd += [
        "-w", "/app",
        "cyborg-agent:latest",
        "python", "agent_base/run_cyborg_coordinator.py"
    ]
    
    # CLI arguments from config
    cmd.extend(["--steps", str(agent_config.get('steps', 30))])

    # Provider/Model
    if agent_config.get('provider'):
        cmd.extend(["--provider", agent_config['provider']])
    if agent_config.get('model'):
        cmd.extend(["--model", agent_config['model']])

    return cmd

def run_instance(args):
    """
    Worker function to run a single evaluation instance.
    args: (instance_id, logs_path, defs_path, connector_logs_path, config, agent_folder)
    """
    instance_id, logs_path, defs_path, connector_logs_path, config, agent_folder = args
    logger.info(f"Starting Instance {instance_id}...")
    
    log_file = os.path.join(os.path.dirname(logs_path), "docker.log")
    
    try:
        num_evals = config.get('num_evaluation_runs', 0)
        if num_evals == 0:
            logger.warning(f"Instance {instance_id}: num_evaluation_runs is 0. Nothing to run.")
            return instance_id, True, None

        logger.info(f"Instance {instance_id} starting {num_evals} evaluation run(s)...")

        # Prepare a config with continual_learning explicitly off
        eval_config = config.copy()
        eval_config['agent_config'] = config['agent_config'].copy()
        eval_config['agent_config']['continual_learning'] = False

        for i in range(1, num_evals + 1):
            logger.info(f"Instance {instance_id} Evaluation Run {i}/{num_evals}")
            eval_cmd = construct_docker_command(
                instance_id, logs_path, defs_path, connector_logs_path, eval_config, agent_folder
            )
            mode = "w" if i == 1 else "a"
            with open(log_file, mode) as f:
                if i > 1:
                    f.write(f"\n\n--- EVALUATION RUN {i} ---\n")
                    f.flush()
                subprocess.run(eval_cmd, check=True, stdout=f, stderr=subprocess.STDOUT)

        logger.info(f"Instance {instance_id} all evaluation runs completed successfully.")
        return instance_id, True, None
    except subprocess.CalledProcessError as e:
        logger.error(f"Instance {instance_id} failed. Check {log_file}")
        return instance_id, False, str(e)

def consolidate_logs(workspaces_dir, aggregated_dir):
    """
    Copies logs from workspaces to the aggregated directory.
    """
    logger.info("Consolidating logs...")
    instance_dirs = glob(os.path.join(workspaces_dir, "instance_*"))
    
    for inst_dir in instance_dirs:
        instance_name = os.path.basename(inst_dir)
        source_logs = os.path.join(inst_dir, "logs")
        target_logs = os.path.join(aggregated_dir, instance_name)
        
        if os.path.exists(source_logs):
            shutil.copytree(source_logs, target_logs)


def parse_connector_log(file_path):
    """
    Parses a connector log file to find the final session summary.
    Returns a dictionary of {provider: {'prompt': int, 'completion': int}}
    """
    usage_data = {}
    
    try:
        with open(file_path, 'r', errors='ignore') as f:
            content = f.read()
            
            # Find the Summary Block using regex
            matches = list(re.finditer(r"--- LLM Connector Session Summary ---\n(.*?)\n-{20,}", content, re.DOTALL))
            
            if not matches:
                return {}
            
            # Take the last summary found in the file
            last_summary_block = matches[-1].group(1)
            lines = last_summary_block.strip().split('\n')
            
            for line in lines:
                if "Provider" in line and "Prompt Tokens" in line:
                    continue
                
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3:
                    provider = parts[0]
                    try:
                        prompt_tokens = int(parts[1].replace(',', ''))
                        completion_tokens = int(parts[2].replace(',', ''))
                        
                        if provider not in usage_data:
                            usage_data[provider] = {'prompt': 0, 'completion': 0}
                        
                        usage_data[provider]['prompt'] += prompt_tokens
                        usage_data[provider]['completion'] += completion_tokens
                    except ValueError:
                        continue
                        
    except Exception as e:
        logger.warning(f"Error parsing connector log {file_path}: {e}")
        
    return usage_data


def calculate_token_costs(aggregated_dir):
    """
    Calculates token usage from connector logs in the aggregated directory.
    Returns a dict with total, average, and per-instance stats.
    """
    instance_stats = []
    total_prompt = 0
    total_completion = 0
    
    # Find connector directories in aggregated logs
    connector_dirs = glob(os.path.join(aggregated_dir, "instance_*", "connector"))
    
    for conn_dir in sorted(connector_dirs):
        log_files = glob(os.path.join(conn_dir, "*.log"))
        if not log_files:
            continue
        
        # Use the earliest log file (sorted alphabetically by timestamp in filename)
        log_files.sort()
        target_log = log_files[0]
        
        # Determine instance name from path
        instance_name = "Unknown"
        path_parts = target_log.split(os.sep)
        for p in path_parts:
            if p.startswith("instance_"):
                instance_name = p
                break
        
        usage = parse_connector_log(target_log)
        
        inst_prompt = 0
        inst_completion = 0
        
        for prov, counts in usage.items():
            inst_prompt += counts['prompt']
            inst_completion += counts['completion']
        
        inst_total = inst_prompt + inst_completion
        
        instance_stats.append({
            "instance": instance_name,
            "prompt": inst_prompt,
            "completion": inst_completion,
            "total": inst_total
        })
        
        total_prompt += inst_prompt
        total_completion += inst_completion
    
    grand_total = total_prompt + total_completion
    num_instances = len(instance_stats)
    
    return {
        "total_prompt": total_prompt,
        "total_completion": total_completion,
        "grand_total": grand_total,
        "num_instances": num_instances,
        "avg_prompt": int(total_prompt / num_instances) if num_instances > 0 else 0,
        "avg_completion": int(total_completion / num_instances) if num_instances > 0 else 0,
        "avg_total": int(grand_total / num_instances) if num_instances > 0 else 0,
        "instance_stats": instance_stats
    }


def process_run_data(file_path, group_list, is_eval=False):
    """
    Helper to extract run data from either a results.json (learning) or a log file (eval).
    """
    session_dir = os.path.dirname(file_path)

    data = {}
    timestamp = ""

    if is_eval:
        data = {'total_attempts': 1}
        folder_name = os.path.basename(session_dir)
        if folder_name.startswith("run_"):
            timestamp = folder_name[4:]
    else:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                timestamp = data.get('session_info', {}).get('session_start_time', '')
        except Exception:
            pass

    # Reward Parsing
    reward_val = None
    try:
        if is_eval:
            log_file = file_path
        else:
            log_files = glob(os.path.join(session_dir, "*_console_mirror.log"))
            log_file = log_files[0] if log_files else None

        if log_file and os.path.exists(log_file):
            with open(log_file, "r") as f:
                content = f.read()
                patterns = [
                    r"Final Evaluation Reward:\s+([\-\d\.]+)",
                    r"Total Cumulative Reward:\s+([\-\d\.]+)",
                    r"Average Reward:\s+([\-\d\.]+)"
                ]
                for pat in patterns:
                    match = re.search(pat, content)
                    if match:
                        reward_val = float(match.group(1))
                        break
    except Exception:
        pass

    run_data = {
        "Attempts": data.get('total_attempts', 1),
        "Final Reward": reward_val,
        "Timestamp": timestamp
    }
    group_list.append(run_data)


def generate_report(aggregated_dir, output_file="summary.md"):
    logger.info(f"Generating report from {aggregated_dir}...")

    instance_groups = {}

    instance_dirs = glob(os.path.join(aggregated_dir, "instance_*"))
    for inst_dir in sorted(instance_dirs):
        inst_name = os.path.basename(inst_dir)
        if inst_name not in instance_groups:
            instance_groups[inst_name] = []

        eval_glob = os.path.join(inst_dir, "runs", "evaluating", "run_*", "*_console_mirror.log")
        for log_file in sorted(glob(eval_glob)):
            process_run_data(log_file, instance_groups[inst_name], is_eval=True)

    report_rows = []
    all_rewards = []

    for inst in sorted(instance_groups.keys()):
        runs = instance_groups[inst]
        if not runs:
            continue

        eval_rewards = [r["Final Reward"] for r in runs if r["Final Reward"] is not None]
        eval_avg_reward = sum(eval_rewards) / len(eval_rewards) if eval_rewards else None
        all_rewards.extend(eval_rewards)

        avg_str = f"{eval_avg_reward:.2f}" if eval_avg_reward is not None else "-"
        rewards_str = ", ".join(f"{r:.2f}" for r in eval_rewards) if eval_rewards else "-"
        row = f"| {inst} | {len(runs)} | {rewards_str} | {avg_str} |"
        report_rows.append(row)

    num_insts = len(instance_groups)
    grand_avg = sum(all_rewards) / len(all_rewards) if all_rewards else None
    grand_avg_str = f"{grand_avg:.2f}" if grand_avg is not None else "-"

    lines = []
    lines.append(f"# Experiment Summary: {os.path.basename(aggregated_dir)}")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- **Total Instances:** {num_insts}")
    lines.append(f"- **Average Reward (all runs):** {grand_avg_str}")
    lines.append("")
    lines.append("## Detailed Results")
    lines.append("| Instance | Eval Runs | Rewards | Avg Reward |")
    lines.append("|---|---|---|---|")
    lines.extend(report_rows)
    
    # Add Token Cost Section
    lines.append("")
    lines.append("## Token Usage")
    token_stats = calculate_token_costs(aggregated_dir)
    if token_stats["num_instances"] > 0:
        lines.append(f"- **Total Prompt Tokens:** {token_stats['total_prompt']:,}")
        lines.append(f"- **Total Completion Tokens:** {token_stats['total_completion']:,}")
        lines.append(f"- **Grand Total Tokens:** {token_stats['grand_total']:,}")
        lines.append(f"- **Avg Tokens per Instance:** {token_stats['avg_total']:,}")
    else:
        lines.append("- No token usage data found.")
    
    with open(os.path.join(aggregated_dir, output_file), "w") as f:
        f.write("\n".join(lines))
    
    logger.info(f"Report generated at {os.path.join(aggregated_dir, output_file)}")


def generate_evaluation_report(aggregated_dir, output_file="evaluation_report.md"):
    """
    Generates a detailed evaluation report tabulating all evaluation rewards.
    """
    logger.info(f"Generating evaluation report from {aggregated_dir}...")

    eval_data = {}
    all_rewards = []

    instance_dirs = glob(os.path.join(aggregated_dir, "instance_*"))
    for inst_dir in sorted(instance_dirs):
        inst_name = os.path.basename(inst_dir)
        eval_data[inst_name] = []

        eval_glob = os.path.join(inst_dir, "runs", "evaluating", "run_*", "*_console_mirror.log")
        for run_idx, log_file in enumerate(sorted(glob(eval_glob)), 1):
            reward_val = None
            try:
                with open(log_file, "r") as f:
                    content = f.read()
                    patterns = [
                        r"Final Evaluation Reward:\s+([\-\d\.]+)",
                        r"Total Cumulative Reward:\s+([\-\d\.]+)",
                        r"Average Reward:\s+([\-\d\.]+)"
                    ]
                    for pat in patterns:
                        match = re.search(pat, content)
                        if match:
                            reward_val = float(match.group(1))
                            break
            except Exception:
                pass

            eval_data[inst_name].append({"run": run_idx, "reward": reward_val})
            if reward_val is not None:
                all_rewards.append(reward_val)

    lines = []
    lines.append(f"# Evaluation Report: {os.path.basename(os.path.dirname(aggregated_dir))}")
    lines.append("")
    lines.append("## Summary Statistics")

    if all_rewards:
        lines.append(f"- **Total Evaluation Runs:** {len(all_rewards)}")
        lines.append(f"- **Average Reward:** {sum(all_rewards) / len(all_rewards):.2f}")
        lines.append(f"- **Min Reward:** {min(all_rewards):.2f}")
        lines.append(f"- **Max Reward:** {max(all_rewards):.2f}")
    else:
        lines.append("- No evaluation data found.")

    lines.append("")
    lines.append("## Per-Instance Evaluation Rewards")
    lines.append("")

    max_runs = max((len(runs) for runs in eval_data.values()), default=0)

    if max_runs > 0:
        header = "| Instance |"
        separator = "|---|"
        for i in range(1, max_runs + 1):
            header += f" Run {i} |"
            separator += "---|"
        header += " Avg |"
        separator += "---|"

        lines.append(header)
        lines.append(separator)

        for inst in sorted(eval_data.keys()):
            runs = eval_data[inst]
            row = f"| {inst} |"
            inst_rewards = []

            for i in range(max_runs):
                if i < len(runs) and runs[i]["reward"] is not None:
                    row += f" {runs[i]['reward']:.2f} |"
                    inst_rewards.append(runs[i]["reward"])
                else:
                    row += " - |"

            row += f" {sum(inst_rewards)/len(inst_rewards):.2f} |" if inst_rewards else " - |"
            lines.append(row)
    else:
        lines.append("No evaluation runs found.")

    report_path = os.path.join(aggregated_dir, output_file)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Evaluation report generated at {report_path}")


def get_instance_eval_reward(instance_dir):
    """
    Extracts the best evaluation reward from an instance's logs.
    Used to select the best instance for incremental learning.
    """
    log_patterns = [
        os.path.join(instance_dir, "logs", "runs", "learning", "*", "*_console_mirror.log"),
        os.path.join(instance_dir, "runs", "learning", "*", "*_console_mirror.log"),  # aggregated_logs structure
    ]
    
    best_reward = -999.0
    
    for pattern in log_patterns:
        for log_file in glob(pattern):
            try:
                with open(log_file, "r") as f:
                    content = f.read()
                    patterns = [
                        r"Final Evaluation Reward:\s+([\-\d\.]+)",
                        r"Total Cumulative Reward:\s+([\-\d\.]+)",
                    ]
                    for pat in patterns:
                        match = re.search(pat, content)
                        if match:
                            reward = float(match.group(1))
                            if reward > best_reward:
                                best_reward = reward
                            break
            except Exception:
                pass
    
    return best_reward


def select_best_instance(workspaces_dir):
    """
    Finds the instance with the best evaluation reward.
    Returns the path to that instance's definitions folder.
    """
    instance_dirs = glob(os.path.join(workspaces_dir, "instance_*"))
    
    best_reward = -999.0
    best_instance_defs = None
    best_instance_name = None
    
    for inst_dir in instance_dirs:
        reward = get_instance_eval_reward(inst_dir)
        if reward > best_reward:
            best_reward = reward
            best_instance_defs = os.path.join(inst_dir, "definitions")
            best_instance_name = os.path.basename(inst_dir)
    
    if best_instance_defs:
        logger.info(f"Selected best instance: {best_instance_name} (reward: {best_reward:.2f})")
    
    return best_instance_defs, best_reward


def generate_incremental_report(master_dir, stage_results, config, graduated_instances=None):
    """
    Generates a comprehensive report covering all instances across all stages.
    """
    logger.info("Generating incremental summary report...")
    
    incremental_config = config.get('incremental', {})
    num_stages = len(stage_results)
    num_instances = config.get('num_instances', 1)
    graduation_threshold = incremental_config.get('graduation_threshold', None)
    graduated_instances = graduated_instances or {}
    
    lines = []
    lines.append("# Incremental Learning Summary")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- **Stages:** {num_stages}")
    lines.append(f"- **Instances per Stage:** {num_instances}")
    lines.append(f"- **Transfer Strategy:** {incremental_config.get('transfer_strategy', 'best')}")
    lines.append(f"- **Attempts per Stage:** {config.get('agent_config', {}).get('max_attempts', 'N/A')}")
    if graduation_threshold:
        lines.append(f"- **Graduation Threshold:** {graduation_threshold}")
        lines.append(f"- **Graduated Instances:** {len(graduated_instances)}/{num_instances}")
    lines.append("")
    
    # Stage summary table
    lines.append("## Stage Summary")
    lines.append("| Stage | Best Instance | Best Reward |")
    lines.append("|-------|---------------|-------------|")
    
    for stage_num, stage_info in stage_results.items():
        best_inst = stage_info.get('best_instance', 'N/A')
        best_reward = stage_info.get('best_reward', -999)
        reward_str = f"{best_reward:.2f}" if best_reward > -999 else "N/A"
        lines.append(f"| {stage_num} | {best_inst} | {reward_str} |")
    
    lines.append("")
    
    # All instances × all stages table
    lines.append("## All Instances × All Stages")
    
    header = "| Instance |"
    separator = "|---|"
    for s in range(1, num_stages + 1):
        header += f" S{s} |"
        separator += "---|"
    lines.append(header)
    lines.append(separator)
    
    # Collect all instance rewards per stage
    all_instances = set()
    stage_data = {}
    
    for stage_num, stage_info in stage_results.items():
        workspaces_dir = stage_info.get('workspaces_dir')
        if not workspaces_dir:
            continue
        
        stage_data[stage_num] = {}
        for inst_dir in glob(os.path.join(workspaces_dir, "instance_*")):
            inst_name = os.path.basename(inst_dir)
            all_instances.add(inst_name)
            reward = get_instance_eval_reward(inst_dir)
            stage_data[stage_num][inst_name] = reward
    
    for inst_name in sorted(all_instances):
        # Extract instance number for graduation lookup
        inst_num = int(inst_name.replace('instance_', '')) if 'instance_' in inst_name else None
        row = f"| {inst_name} |"
        for stage_num in range(1, num_stages + 1):
            if stage_num in stage_data and inst_name in stage_data[stage_num]:
                reward = stage_data[stage_num][inst_name]
                # Mark graduated instances
                is_graduated = (inst_num in graduated_instances and graduated_instances[inst_num]['stage'] == stage_num)
                if is_graduated:
                    row += f" **{reward:.1f}** 🎓 |"
                elif reward > -1.1:
                    row += f" **{reward:.1f}** |"
                else:
                    row += f" {reward:.1f} |"
            else:
                row += " - |"
        lines.append(row)
    
    # Graduation summary section
    if graduation_threshold and graduated_instances:
        lines.append("")
        lines.append("## Graduation Summary")
        lines.append("")
        lines.append(f"> **Threshold:** Instances with reward > {graduation_threshold} stop learning.")
        lines.append("")
        lines.append("| Instance | Graduated Stage | Final Reward |")
        lines.append("|----------|-----------------|--------------|")
        for inst_num in sorted(graduated_instances.keys()):
            info = graduated_instances[inst_num]
            lines.append(f"| instance_{inst_num} | Stage {info['stage']} | {info['reward']:.2f} |")
        
        # List ungraduated instances
        ungraduated = [i for i in range(1, num_instances + 1) if i not in graduated_instances]
        if ungraduated:
            lines.append("")
            lines.append(f"**Not Graduated:** {', '.join(f'instance_{i}' for i in ungraduated)}")
    
    # Token cost aggregation across all stages
    lines.append("")
    lines.append("## Token Usage")
    lines.append("")
    
    total_prompt_all = 0
    total_completion_all = 0
    stage_token_data = []
    
    for stage_num, stage_info in stage_results.items():
        aggregated_dir = stage_info.get('aggregated_dir')
        if aggregated_dir and os.path.exists(aggregated_dir):
            try:
                token_stats = calculate_token_costs(aggregated_dir)
                stage_prompt = token_stats.get('total_prompt', 0)
                stage_completion = token_stats.get('total_completion', 0)
                stage_total = token_stats.get('grand_total', 0)
                total_prompt_all += stage_prompt
                total_completion_all += stage_completion
                stage_token_data.append({
                    'stage': stage_num,
                    'prompt': stage_prompt,
                    'completion': stage_completion,
                    'total': stage_total
                })
            except Exception as e:
                logger.warning(f"Could not calculate token costs for stage {stage_num}: {e}")
    
    # Check final_evaluation directory if it exists
    final_eval_dir = os.path.join(master_dir, "final_evaluation", "aggregated_logs")
    if os.path.exists(final_eval_dir):
        try:
            token_stats = calculate_token_costs(final_eval_dir)
            eval_prompt = token_stats.get('total_prompt', 0)
            eval_completion = token_stats.get('total_completion', 0)
            eval_total = token_stats.get('grand_total', 0)
            total_prompt_all += eval_prompt
            total_completion_all += eval_completion
            stage_token_data.append({
                'stage': 'Final Eval',
                'prompt': eval_prompt,
                'completion': eval_completion,
                'total': eval_total
            })
        except Exception as e:
            logger.warning(f"Could not calculate token costs for final evaluation: {e}")
    
    grand_total_all = total_prompt_all + total_completion_all
    
    if stage_token_data:
        lines.append("| Phase | Prompt Tokens | Completion Tokens | Total |")
        lines.append("|-------|---------------|-------------------|-------|")
        for d in stage_token_data:
            lines.append(f"| Stage {d['stage']} | {d['prompt']:,} | {d['completion']:,} | {d['total']:,} |")
        lines.append(f"| **TOTAL** | **{total_prompt_all:,}** | **{total_completion_all:,}** | **{grand_total_all:,}** |")
    else:
        lines.append("*Token usage data not available.*")
    
    # Write report
    report_path = os.path.join(master_dir, "incremental_summary.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    
    logger.info(f"Incremental report generated at {report_path}")


import signal


def regenerate_incremental_report(experiment_path):
    """
    Regenerates incremental_summary.md for an existing multi-stage experiment.
    Scans for stage_* directories and rebuilds the report.
    """
    experiment_path = os.path.abspath(experiment_path)
    
    if not os.path.exists(experiment_path):
        logger.error(f"Experiment path not found: {experiment_path}")
        return
    
    logger.info(f"Regenerating incremental report for: {experiment_path}")
    
    # Find stage directories
    stage_dirs = sorted(glob(os.path.join(experiment_path, "stage_*")))
    if not stage_dirs:
        logger.error(f"No stage_* directories found in {experiment_path}")
        return
    
    # Load config from experiment if exists
    config_path = os.path.join(experiment_path, "experiment_config.yaml")
    if os.path.exists(config_path):
        config = parse_config(config_path)
    else:
        logger.warning("No experiment_config.yaml found, using defaults")
        config = {'num_instances': 1, 'incremental': {}}
    
    # Build stage_results from existing directories
    stage_results = {}
    for stage_dir in stage_dirs:
        stage_name = os.path.basename(stage_dir)
        try:
            stage_num = int(stage_name.replace("stage_", ""))
        except ValueError:
            continue
        
        workspaces_dir = os.path.join(stage_dir, "workspaces")
        aggregated_dir = os.path.join(stage_dir, "aggregated_logs")
        
        # Find best instance
        best_defs, best_reward = None, -999
        if os.path.exists(workspaces_dir):
            best_defs, best_reward = select_best_instance(workspaces_dir)
        
        best_instance_name = "N/A"
        if best_defs:
            best_instance_name = os.path.basename(os.path.dirname(best_defs))
        
        stage_results[stage_num] = {
            'workspaces_dir': workspaces_dir,
            'aggregated_dir': aggregated_dir,
            'best_instance': best_instance_name,
            'best_reward': best_reward
        }
    
    logger.info(f"Found {len(stage_results)} stages")
    
    # Generate report (graduated_instances not available when regenerating)
    generate_incremental_report(experiment_path, stage_results, config, graduated_instances={})
    
    logger.info("Incremental report regeneration complete.")


def resume_incremental_experiment(config, args):
    """
    Resumes a failed incremental experiment from a given stage.
    Deletes stage_N onwards (and final_evaluation) then re-runs from stage N,
    using each instance's definitions from stage N-1 as the starting point.
    Stages 1..N-1 are left completely untouched.
    """
    resume_path = os.path.abspath(args.resume)
    resume_from = args.resume_from_stage

    if not os.path.exists(resume_path):
        logger.error(f"Resume path not found: {resume_path}")
        return

    # --- Load config from the experiment dir (ground truth), override with CLI config only for
    #     fields the user may have changed (e.g. max_parallel_workers, num_evaluation_runs).
    saved_config_path = os.path.join(resume_path, "experiment_config.yaml")
    if os.path.exists(saved_config_path):
        saved_config = parse_config(saved_config_path)
        logger.info(f"Loaded saved config from {saved_config_path}")
    else:
        saved_config = config
        logger.warning("No experiment_config.yaml in experiment dir — using CLI config")

    # Allow CLI config to override runtime-only keys
    for key in ("max_parallel_workers", "num_evaluation_runs"):
        if key in config:
            saved_config[key] = config[key]

    config = saved_config
    incremental_config = config.get('incremental', {})
    num_stages      = incremental_config.get('stages', 1)
    transfer_strategy = incremental_config.get('transfer_strategy', 'best')
    agent_folder    = config.get('agent_folder', 'agent_base')
    num_instances   = config.get('num_instances', 1)
    final_num_evals = config.get('num_evaluation_runs', 0)
    graduation_threshold = incremental_config.get('graduation_threshold', None)

    if resume_from < 1 or resume_from > num_stages:
        logger.error(f"--resume-from-stage {resume_from} is out of range (1..{num_stages})")
        return

    # --- Extract timestamp from the experiment directory name so the signal
    #     handler uses the same container-name prefix as the original run.
    dir_basename = os.path.basename(resume_path)
    # Format: {exp_name}_{YYYYMMDD_HHMMSS}  — last two underscore-joined tokens are the timestamp
    parts = dir_basename.rsplit('_', 2)
    if len(parts) == 3 and len(parts[1]) == 8 and len(parts[2]) == 6:
        timestamp = f"{parts[1]}_{parts[2]}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.warning(f"Could not parse timestamp from dir name — using new timestamp: {timestamp}")

    logger.info(f"Resuming experiment: {resume_path}")
    logger.info(f"Resuming from stage {resume_from} (stages 1..{resume_from - 1} are preserved)")

    # --- Delete only the failed stages and final_evaluation --------------------
    for stale_stage in range(resume_from, num_stages + 1):
        stale_dir = os.path.join(resume_path, f"stage_{stale_stage}")
        if os.path.exists(stale_dir):
            logger.info(f"Removing failed stage dir: {stale_dir}")
            shutil.rmtree(stale_dir)

    final_eval_dir = os.path.join(resume_path, "final_evaluation")
    if os.path.exists(final_eval_dir):
        logger.info(f"Removing failed final_evaluation dir: {final_eval_dir}")
        shutil.rmtree(final_eval_dir)

    # --- Reconstruct stage_results from existing completed stages --------------
    stage_results = {}
    graduated_instances = {}  # Not persisted across runs; skip graduation re-check for past stages
    prev_workspaces_dir = None

    for s in range(1, resume_from):
        stage_dir      = os.path.join(resume_path, f"stage_{s}")
        workspaces_dir = os.path.join(stage_dir, "workspaces")
        aggregated_dir = os.path.join(stage_dir, "aggregated_logs")

        if not os.path.exists(workspaces_dir):
            logger.warning(f"Stage {s} workspaces not found at {workspaces_dir} — skipping reconstruction")
            continue

        best_defs, best_reward = select_best_instance(workspaces_dir)
        best_instance_name = "N/A"
        if best_defs:
            best_instance_name = os.path.basename(os.path.dirname(best_defs))

        stage_results[s] = {
            'workspaces_dir': workspaces_dir,
            'aggregated_dir': aggregated_dir,
            'best_instance':  best_instance_name,
            'best_reward':    best_reward,
        }
        prev_workspaces_dir = workspaces_dir
        logger.info(f"Reconstructed stage {s}: best={best_instance_name} reward={best_reward:.2f}")

    # For "best" strategy we need to know which definitions to start stage resume_from from.
    # For "individual" strategy each instance reads its own defs from prev_workspaces_dir.
    current_definitions = os.path.abspath(f"{agent_folder}/agents/prompts/definitions")
    if transfer_strategy == 'best' and prev_workspaces_dir:
        best_defs, _ = select_best_instance(prev_workspaces_dir)
        if best_defs:
            current_definitions = best_defs

    # --- Signal handler --------------------------------------------------------
    setup_signal_handler(timestamp)
    config['experiment_timestamp'] = timestamp

    # --- Resume stage loop (stage resume_from .. num_stages) -------------------
    for stage_num in range(resume_from, num_stages + 1):
        # Graduation guard
        if graduation_threshold:
            remaining = num_instances - len(graduated_instances)
            if remaining == 0:
                logger.info(f"All instances graduated — stopping at stage {stage_num - 1}.")
                break
            logger.info(f"Stage {stage_num}: {remaining} instances remaining ({len(graduated_instances)} graduated)")

        logger.info(f"\n{'='*60}\n STAGE {stage_num}/{num_stages}  [RESUMED]\n{'='*60}")

        stage_dir      = os.path.join(resume_path, f"stage_{stage_num}")
        workspaces_dir = os.path.join(stage_dir, "workspaces")
        aggregated_dir = os.path.join(stage_dir, "aggregated_logs")
        os.makedirs(workspaces_dir, exist_ok=True)
        os.makedirs(aggregated_dir, exist_ok=True)

        stage_config = config.copy()
        stage_config['num_evaluation_runs'] = 0
        stage_config['experiment_timestamp'] = timestamp

        tasks = []
        active_instances = []
        for i in range(1, num_instances + 1):
            if graduation_threshold and i in graduated_instances:
                logger.info(f"Instance {i} already graduated — skipping.")
                continue

            active_instances.append(i)

            if transfer_strategy == "individual" and prev_workspaces_dir:
                prev_inst_defs = os.path.join(prev_workspaces_dir, f"instance_{i}", "definitions")
                inst_defs_source = prev_inst_defs if os.path.exists(prev_inst_defs) else current_definitions
            else:
                inst_defs_source = current_definitions

            inst_dir, logs_path, defs_path, connector_logs_path = prepare_instance(
                i, workspaces_dir, inst_defs_source, agent_folder
            )
            tasks.append((i, logs_path, defs_path, connector_logs_path, stage_config, agent_folder, False))

        if not tasks:
            logger.info(f"No instances to run in stage {stage_num}.")
            continue

        # Snapshot initial definitions
        for i in active_instances:
            inst_defs         = os.path.join(workspaces_dir, f"instance_{i}", "definitions")
            inst_defs_initial = os.path.join(workspaces_dir, f"instance_{i}", "definitions_initial")
            if os.path.exists(inst_defs):
                if os.path.exists(inst_defs_initial):
                    shutil.rmtree(inst_defs_initial)
                shutil.copytree(inst_defs, inst_defs_initial)
        logger.info(f"Stage {stage_num}: Saved initial definitions snapshot for {len(active_instances)} instances.")

        max_workers = config.get('max_parallel_workers', 2)
        logger.info(f"Stage {stage_num}: Launching {len(active_instances)} instances...")

        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(run_instance, tasks))
        except KeyboardInterrupt:
            logger.warning("Interrupted during stage execution.")
            return

        consolidate_logs(workspaces_dir, aggregated_dir)
        generate_report(aggregated_dir)

        # Graduation check
        if graduation_threshold:
            for i in active_instances:
                inst_dir = os.path.join(workspaces_dir, f"instance_{i}")
                reward = get_instance_eval_reward(inst_dir)
                if reward > graduation_threshold:
                    graduated_instances[i] = {
                        'stage': stage_num, 'reward': reward,
                        'definitions': os.path.join(inst_dir, "definitions")
                    }
                    logger.info(f"Instance {i} GRADUATED at stage {stage_num} with reward {reward:.2f}")

        best_defs, best_reward = select_best_instance(workspaces_dir)
        best_instance_name = "N/A"
        if best_defs:
            best_instance_name = os.path.basename(os.path.dirname(best_defs))
            if transfer_strategy == "best":
                current_definitions = best_defs

        stage_results[stage_num] = {
            'workspaces_dir': workspaces_dir,
            'aggregated_dir': aggregated_dir,
            'best_instance':  best_instance_name,
            'best_reward':    best_reward,
            'active_instances': active_instances,
            'graduated_this_stage': [
                i for i in active_instances
                if i in graduated_instances and graduated_instances[i]['stage'] == stage_num
            ]
        }

        prev_workspaces_dir = workspaces_dir
        logger.info(f"Stage {stage_num} complete. Best: {best_instance_name} ({best_reward:.2f}).")

    # --- Final evaluation ------------------------------------------------------
    if final_num_evals > 0:
        logger.info(f"\n{'='*60}\n FINAL EVALUATION ({final_num_evals} runs)\n{'='*60}")

        eval_dir        = os.path.join(resume_path, "final_evaluation")
        eval_workspaces = os.path.join(eval_dir, "workspaces")
        eval_aggregated = os.path.join(eval_dir, "aggregated_logs")
        os.makedirs(eval_workspaces, exist_ok=True)
        os.makedirs(eval_aggregated, exist_ok=True)

        eval_config = config.copy()
        eval_config['agent_config'] = config['agent_config'].copy()
        eval_config['agent_config']['continual_learning'] = False
        eval_config['num_evaluation_runs'] = final_num_evals
        eval_config['experiment_timestamp'] = timestamp

        eval_tasks = []
        for i in range(1, num_instances + 1):
            if i in graduated_instances:
                eval_defs_source = graduated_instances[i]['definitions']
                logger.info(f"Instance {i}: Using definitions from graduation stage {graduated_instances[i]['stage']}")
            elif transfer_strategy == "individual" and prev_workspaces_dir:
                eval_defs_source = os.path.join(prev_workspaces_dir, f"instance_{i}", "definitions")
                if not os.path.exists(eval_defs_source):
                    eval_defs_source = current_definitions
            else:
                eval_defs_source = current_definitions

            inst_dir, logs_path, defs_path, connector_logs_path = prepare_instance(
                i, eval_workspaces, eval_defs_source, agent_folder
            )
            eval_tasks.append((i, logs_path, defs_path, connector_logs_path, eval_config, agent_folder, True))

        try:
            with ProcessPoolExecutor(max_workers=config.get('max_parallel_workers', 2)) as executor:
                list(executor.map(run_instance, eval_tasks))
        except KeyboardInterrupt:
            logger.warning("Interrupted during final evaluation.")

        consolidate_logs(eval_workspaces, eval_aggregated)
        generate_report(eval_aggregated)
        generate_evaluation_report(eval_aggregated)

    # --- Regenerate combined report --------------------------------------------
    generate_incremental_report(resume_path, stage_results, config, graduated_instances)

    logger.info(f"\n{'='*60}\n RESUME COMPLETE\n{'='*60}")
    logger.info(f"Results at: {resume_path}")


def build_docker_image(tag="cyborg-agent:latest"):
    """Builds the Docker image from the local Dockerfile."""
    logger.info(f"Building Docker image '{tag}'...")
    cmd = ["docker", "build", "-t", tag, "."]
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Docker image '{tag}' built successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Docker build failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Run CybORG Agent Evaluation Experiments with Docker")
    parser.add_argument("--config", type=str, default="experiment_config.yaml", help="Path to config file")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the Docker image before running")
    parser.add_argument("--report-only", action="store_true", help="Only generate report from existing data")
    parser.add_argument("--report-path", type=str, help="Path to aggregated_logs directory (requires --report-only)")
    parser.add_argument("--progress", action="store_true",
                        help="Show a live per-instance progress dashboard. "
                             "Reads docker.log for each instance. Requires 'rich' for in-place display; "
                             "falls back to periodic logger output otherwise.")
    args = parser.parse_args()

    # Load environment variables from .env
    load_env_file()

    if args.rebuild:
        build_docker_image()

    if args.report_only:
        if args.report_path:
            logger.info(f"Generating report for existing data at {args.report_path}...")
            generate_report(args.report_path)
            generate_evaluation_report(args.report_path)
            logger.info("Report generation complete.")
        else:
            logger.error("Please provide --report-path when using --report-only")
        return

    config = parse_config(args.config)
    run_regular_experiment(config, args)


def run_regular_experiment(config, args):
    """Runs a standard evaluation experiment."""
    exp_name = config.get('experiment_name', 'experiment')
    num_instances = config.get('num_instances', 1)
    num_evals = config.get('num_evaluation_runs', 0)

    if num_evals == 0:
        logger.error("num_evaluation_runs must be > 0. Exiting.")
        return

    # 1. Setup Dirs
    base_dir, workspaces_dir, aggregated_dir, timestamp = setup_experiment_dirs(exp_name)
    logger.info(f"Initialized Experiment: {base_dir}")
    config['experiment_timestamp'] = timestamp

    setup_signal_handler(timestamp)

    # 2. Resolve definitions source
    agent_folder = config.get('agent_folder', 'agent_base')
    definitions_source_config = config.get('definitions_source')
    if definitions_source_config:
        definitions_source = os.path.abspath(definitions_source_config)
        logger.info(f"Using custom definitions source: {definitions_source}")
    else:
        definitions_source = os.path.abspath(f"{agent_folder}/agents/prompts/definitions")
        logger.info(f"Using default definitions source: {definitions_source}")

    if not os.path.exists(definitions_source):
        logger.error(f"Definitions source not found: {definitions_source}")
        return

    # 3. Prepare instance tasks
    tasks = []
    for i in range(1, num_instances + 1):
        inst_dir, logs_path, defs_path, connector_logs_path = prepare_instance(
            i, workspaces_dir, definitions_source, agent_folder
        )
        tasks.append((i, logs_path, defs_path, connector_logs_path, config, agent_folder))

    # 4. Save config snapshot
    shutil.copy(args.config, os.path.join(base_dir, "experiment_config.yaml"))

    # 5. Run in parallel
    max_workers = config.get('max_parallel_workers', 2)
    total_steps = config.get('agent_config', {}).get('steps', 30)
    logger.info(f"Launching {num_instances} instance(s) with up to {max_workers} workers...")

    # Optional live progress dashboard (--progress flag only)
    monitor = None
    if getattr(args, 'progress', False):
        monitor = ProgressMonitor(
            workspaces_dir=workspaces_dir,
            num_instances=num_instances,
            num_eval_runs=num_evals,
            total_steps=total_steps,
        )
        monitor.start()
        # Expose to signal handler so Ctrl+C can restore the terminal cleanly
        global _active_monitor
        _active_monitor = monitor

    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            # When --progress is active, suppress the worker-process console logger.
            # Workers write all output to docker.log via subprocess stdout redirect.
            # Without this, their logger.info() calls write to inherited stdout and
            # break rich.Live's cursor tracking (causing duplicate table renders).
            initializer=_worker_suppress_logging if monitor else None,
        ) as executor:
            results = list(executor.map(run_instance, tasks))
    except KeyboardInterrupt:
        if monitor:
            monitor.stop()
        logger.warning("Interrupted. Cleanup handled by signal handler.")
        return

    if monitor:
        monitor.stop()

    # 6. Consolidate & Report
    consolidate_logs(workspaces_dir, aggregated_dir)
    generate_report(aggregated_dir)
    generate_evaluation_report(aggregated_dir)
    logger.info(f"Experiment complete. Results at: {base_dir}")


def run_incremental_experiment(config, args):
    """Runs an incremental (multi-stage) experiment."""
    exp_name = config.get('experiment_name', 'experiment')
    num_instances = config.get('num_instances', 1)
    incremental_config = config.get('incremental', {})
    num_stages = incremental_config.get('stages', 1)
    transfer_strategy = incremental_config.get('transfer_strategy', 'best')
    agent_folder = config.get('agent_folder', 'agent_base')
    final_num_evals = config.get('num_evaluation_runs', 0)
    
    # 1. Setup master directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_dir = f"experiments/{exp_name}_{timestamp}"
    os.makedirs(master_dir, exist_ok=True)
    
    logger.info(f"Initialized Incremental Experiment: {master_dir}")
    logger.info(f"Stages: {num_stages}, Transfer Strategy: {transfer_strategy}")
    
    # Save config
    shutil.copy(args.config, os.path.join(master_dir, "experiment_config.yaml"))
    
    # Signal handler
    setup_signal_handler(timestamp)
    
    # Get initial definitions source
    definitions_source_config = config.get('definitions_source')
    if definitions_source_config:
        current_definitions = os.path.abspath(definitions_source_config)
    else:
        current_definitions = os.path.abspath(f"{agent_folder}/agents/prompts/definitions")
    
    if not os.path.exists(current_definitions):
        logger.error(f"Definitions source not found: {current_definitions}")
        return
    
    stage_results = {}
    prev_workspaces_dir = None
    
    # Graduation tracking (works for both strategies)
    graduation_threshold = incremental_config.get('graduation_threshold', None)
    graduated_instances = {}  # instance_num -> {'stage': N, 'reward': R, 'definitions': path}
    
    if graduation_threshold:
        logger.info(f"Graduation threshold enabled: {graduation_threshold}")
    
    # If eval-only, skip all learning stages
    eval_only = getattr(args, 'eval_only', False)
    if eval_only:
        logger.info("Eval-only mode: Skipping all learning stages")
        num_stages = 0  # Skip stage loop entirely
    
    # Run stages
    for stage_num in range(1, num_stages + 1):
        # Check if all instances have graduated
        if graduation_threshold:
            remaining = num_instances - len(graduated_instances)
            if remaining == 0:
                logger.info(f"All instances have graduated! Stopping at stage {stage_num - 1}.")
                break
            logger.info(f"Stage {stage_num}: {remaining} instances remaining ({len(graduated_instances)} graduated)")
        
        logger.info(f"\n{'='*60}\n STAGE {stage_num}/{num_stages}\n{'='*60}")
        
        # Create stage directory
        stage_dir = os.path.join(master_dir, f"stage_{stage_num}")
        workspaces_dir = os.path.join(stage_dir, "workspaces")
        aggregated_dir = os.path.join(stage_dir, "aggregated_logs")
        os.makedirs(workspaces_dir, exist_ok=True)
        os.makedirs(aggregated_dir, exist_ok=True)
        
        # Create stage-specific config (disable per-stage evals)
        stage_config = config.copy()
        stage_config['num_evaluation_runs'] = 0  # Embedded eval only
        stage_config['experiment_timestamp'] = timestamp  # Use master timestamp for cleanup
        
        # Prepare instances (skip graduated ones)
        tasks = []
        active_instances = []
        for i in range(1, num_instances + 1):
            # Skip graduated instances
            if graduation_threshold and i in graduated_instances:
                logger.info(f"Instance {i} already graduated at stage {graduated_instances[i]['stage']}, skipping.")
                continue
            
            active_instances.append(i)
            
            # Determine definitions source based on transfer strategy
            if stage_num == 1:
                inst_defs_source = current_definitions
            elif transfer_strategy == "individual" and prev_workspaces_dir:
                # Each instance uses its own definitions from previous stage
                prev_inst_defs = os.path.join(prev_workspaces_dir, f"instance_{i}", "definitions")
                if os.path.exists(prev_inst_defs):
                    inst_defs_source = prev_inst_defs
                else:
                    inst_defs_source = current_definitions
            else:
                # "best" strategy: all instances use best from previous stage
                inst_defs_source = current_definitions
            
            inst_dir, logs_path, defs_path, connector_logs_path = prepare_instance(
                i, workspaces_dir, inst_defs_source, agent_folder
            )
            tasks.append((i, logs_path, defs_path, connector_logs_path, stage_config, agent_folder, False))
        
        if not tasks:
            logger.info(f"No instances to run in stage {stage_num}.")
            continue
        
        # Snapshot initial definitions before learning mutates them
        for i in active_instances:
            inst_defs = os.path.join(workspaces_dir, f"instance_{i}", "definitions")
            inst_defs_initial = os.path.join(workspaces_dir, f"instance_{i}", "definitions_initial")
            if os.path.exists(inst_defs):
                if os.path.exists(inst_defs_initial):
                    shutil.rmtree(inst_defs_initial)
                shutil.copytree(inst_defs, inst_defs_initial)
        logger.info(f"Stage {stage_num}: Saved initial definitions snapshot for {len(active_instances)} instances.")
        
        # Run stage in parallel
        max_workers = config.get('max_parallel_workers', 2)
        logger.info(f"Stage {stage_num}: Launching {len(active_instances)} instances...")
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(run_instance, tasks))
        except KeyboardInterrupt:
            logger.warning("Interrupted during stage execution.")
            return
        
        # Consolidate and report for this stage
        consolidate_logs(workspaces_dir, aggregated_dir)
        generate_report(aggregated_dir)
        
        # Check for graduation (for both strategies)
        if graduation_threshold:
            for i in active_instances:
                inst_dir = os.path.join(workspaces_dir, f"instance_{i}")
                reward = get_instance_eval_reward(inst_dir)
                if reward > graduation_threshold:
                    graduated_instances[i] = {
                        'stage': stage_num,
                        'reward': reward,
                        'definitions': os.path.join(inst_dir, "definitions")
                    }
                    logger.info(f"Instance {i} GRADUATED at stage {stage_num} with reward {reward:.2f}")
        
        # Select best instance for next stage (if using "best" strategy)
        best_defs, best_reward = select_best_instance(workspaces_dir)
        best_instance_name = "N/A"
        if best_defs:
            best_instance_name = os.path.basename(os.path.dirname(best_defs))
            if transfer_strategy == "best":
                current_definitions = best_defs
        
        stage_results[stage_num] = {
            'workspaces_dir': workspaces_dir,
            'aggregated_dir': aggregated_dir,
            'best_instance': best_instance_name,
            'best_reward': best_reward,
            'active_instances': active_instances,
            'graduated_this_stage': [i for i in active_instances if i in graduated_instances and graduated_instances[i]['stage'] == stage_num]
        }
        
        prev_workspaces_dir = workspaces_dir
        
        graduated_count = len(graduated_instances)
        logger.info(f"Stage {stage_num} complete. Best: {best_instance_name} ({best_reward:.2f}). Graduated: {graduated_count}/{num_instances}")
    
    # Run final evaluation if requested
    if final_num_evals > 0:
        logger.info(f"\n{'='*60}\n FINAL EVALUATION ({final_num_evals} runs)\n{'='*60}")
        
        eval_dir = os.path.join(master_dir, "final_evaluation")
        eval_workspaces = os.path.join(eval_dir, "workspaces")
        eval_aggregated = os.path.join(eval_dir, "aggregated_logs")
        os.makedirs(eval_workspaces, exist_ok=True)
        os.makedirs(eval_aggregated, exist_ok=True)
        
        # Prepare eval config
        eval_config = config.copy()
        eval_config['agent_config'] = config['agent_config'].copy()
        eval_config['agent_config']['continual_learning'] = False
        eval_config['num_evaluation_runs'] = final_num_evals
        eval_config['experiment_timestamp'] = timestamp  # Use master timestamp for cleanup
        
        eval_tasks = []
        for i in range(1, num_instances + 1):
            # For eval, use the definitions from graduation point (if graduated)
            # or from the final stage (if never graduated)
            if i in graduated_instances:
                # Use definitions from the stage where instance graduated
                eval_defs_source = graduated_instances[i]['definitions']
                logger.info(f"Instance {i}: Using definitions from graduation at stage {graduated_instances[i]['stage']}")
            elif transfer_strategy == "individual" and prev_workspaces_dir:
                # Individual: Use definitions from the last stage this instance participated in
                eval_defs_source = os.path.join(prev_workspaces_dir, f"instance_{i}", "definitions")
                if not os.path.exists(eval_defs_source):
                    eval_defs_source = current_definitions
            else:
                # Best strategy: Use current_definitions (best from last stage)
                eval_defs_source = current_definitions
            
            inst_dir, logs_path, defs_path, connector_logs_path = prepare_instance(
                i, eval_workspaces, eval_defs_source, agent_folder
            )
            eval_tasks.append((i, logs_path, defs_path, connector_logs_path, eval_config, agent_folder, True))
        
        try:
            with ProcessPoolExecutor(max_workers=config.get('max_parallel_workers', 2)) as executor:
                list(executor.map(run_instance, eval_tasks))
        except KeyboardInterrupt:
            logger.warning("Interrupted during final evaluation.")
        
        consolidate_logs(eval_workspaces, eval_aggregated)
        generate_report(eval_aggregated)
        generate_evaluation_report(eval_aggregated)
    
    # Generate combined incremental report
    generate_incremental_report(master_dir, stage_results, config, graduated_instances)
    
    logger.info(f"\n{'='*60}\n INCREMENTAL LEARNING COMPLETE\n{'='*60}")
    logger.info(f"Results at: {master_dir}")


def setup_signal_handler(timestamp):
    """Sets up signal handler for graceful shutdown."""
    def cleanup_handler(signum, frame):
        # Stop the progress monitor FIRST so rich.Live releases the terminal
        # before we write any output.  Without this, the cleanup messages are
        # swallowed by the live display and the table doubles on exit.
        global _active_monitor
        if _active_monitor is not None:
            try:
                _active_monitor.stop()
            except Exception:
                pass
            _active_monitor = None

        logger.warning("\n\n!!! RECEIVED SIGNAL TO STOP. CLEANING UP... !!!")
        container_pattern = f"cyborg_worker_{timestamp}_"
        try:
            cmd_list = ["docker", "ps", "-q", "--filter", f"name={container_pattern}"]
            result = subprocess.run(cmd_list, capture_output=True, text=True)
            container_ids = result.stdout.strip().split()
            if container_ids:
                logger.info(f"Stopping {len(container_ids)} containers...")
                subprocess.run(["docker", "stop", "-t", "0"] + container_ids, check=False)
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        sys.exit(1)

    signal.signal(signal.SIGINT, cleanup_handler)
    logger.info("Signal handler registered. Use Ctrl+C to stop.")


if __name__ == "__main__":
    main()

