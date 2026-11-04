"""Wait for the full-data workers, then run GB alone and finalize the comparison."""
import json
import time
import traceback
from pathlib import Path

PROJECT=Path(__file__).resolve().parent


def main():
    from extended_registry import NEW_NEURAL,NEW_CLASSICAL
    from train_adapted import save_json
    progress=PROJECT/'artifacts/expanded_classical/gb_progress.json'
    last_report=0
    while True:
        pending=[]
        for family,models in [('expanded_neural',NEW_NEURAL),('expanded_classical',[m for m in NEW_CLASSICAL if m!='GB'])]:
            pending.extend(m for m in models if not (PROJECT/'artifacts'/family/m/'complete.json').exists())
        if not pending:break
        save_json(progress,dict(stage='waiting_for_other_full_data_models',pending=pending,
                  reason='Run GB after other workers release RAM/GPU resources'))
        if time.monotonic()-last_report>120:
            from report_expanded import main as refresh
            refresh();last_report=time.monotonic()
        time.sleep(5)
    time.sleep(15)
    from train_gradient_boosting import main as train_gb
    train_gb()
    from report_expanded import main as refresh
    refresh()
    from verify_expanded import main as audit
    audit();refresh()
    print('EXTENDED COMPARISON COMPLETE',flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        target=PROJECT/'artifacts/extended_report';target.mkdir(exist_ok=True)
        (target/'finalization_failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
