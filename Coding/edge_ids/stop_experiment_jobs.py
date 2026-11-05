"""Stop this user's project training/generation/combination jobs; delete no data."""
import os
from pathlib import Path
import signal
import time
from cleanup_adversarial_artifacts import SCRIPTS

PROJECT=Path(__file__).resolve().parent
JOBS=set(SCRIPTS)|{'baseline.py','train_adapted.py','train_cpu_classical.py',
    'train_expanded_neural.py','train_gradient_boosting.py','train_linear_kernel.py',
    'complete_remaining.py','finish_expanded.py','combine_training.py'}

def identity(process):
    # starttime protects against terminating a reused PID after the grace period.
    return (process/'stat').read_text().rsplit(')',1)[1].split()[19]

def matching():
    jobs=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            args=(p/'cmdline').read_bytes().split(b'\0');cwd=(p/'cwd').resolve()
            for raw in args[1:]:
                arg=Path(raw.decode(errors='replace'))
                if arg.name not in JOBS:continue
                source=arg if arg.is_absolute() else cwd/arg
                if source.resolve()==PROJECT/arg.name:
                    jobs.append((int(p.name),identity(p),arg.name));break
        except (OSError,PermissionError,IndexError):continue
    return jobs

def main():
    if not Path('/proc').is_dir():raise SystemExit('Run this helper on the Linux lab server.')
    jobs=matching()
    for pid,start,name in jobs:
        try:os.kill(pid,signal.SIGTERM);print('Stopping',pid,name,flush=True)
        except ProcessLookupError:pass
    if jobs:time.sleep(3)
    for pid,start,name in jobs:
        try:
            if identity(Path('/proc')/str(pid))==start:os.kill(pid,signal.SIGKILL)
        except (OSError,IndexError):pass
    time.sleep(.5)
    remaining=matching()
    if remaining:raise SystemExit('Project jobs remain: '+str(remaining))
    print('Verified: no matching training, sample-generation or combination jobs remain. No datasets or results deleted.')

if __name__=='__main__':main()
