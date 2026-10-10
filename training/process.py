"""Stream child output while preserving a complete diagnostic log."""
from collections import deque
from pathlib import Path
import shlex
import subprocess


def run_logged(command, log_path, *, env=None):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print('Running:', shlex.join(list(map(str, command))), flush=True)
    tail = deque(maxlen=35)
    with log_path.open('w', encoding='utf-8') as log:
        with subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                              errors='replace', bufsize=1, env=env) as child:
            for line in child.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
                tail.append(line)
            returncode = child.wait()
    if returncode:
        raise RuntimeError(f'Command failed (exit {returncode}): {shlex.join(list(map(str, command)))}\n'
                           f'Complete log: {log_path}\nLast output:\n' + ''.join(tail))
    return subprocess.CompletedProcess(command, returncode)
