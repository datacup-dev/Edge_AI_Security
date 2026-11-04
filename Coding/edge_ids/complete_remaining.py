"""Run the two remaining jobs sequentially and finalize audited baseline outputs."""
import json
import subprocess
import sys
import traceback
from pathlib import Path

PROJECT=Path(__file__).resolve().parent


def main():
    # Children exit between phases to release GPU/CPU memory before the next job.
    stages=[('gradient_boosting','train_gradient_boosting.py'),
            ('exact_knn','train_cpu_classical.py'),
            ('report','report_expanded.py'),
            ('artifact_audit','verify_expanded.py'),
            ('loader_audit','verify_overall_loader.py'),
            ('verified_report','report_expanded.py'),
            ('notebook','update_expanded_preview.py'),
            ('final_report','report_expanded.py')]
    output=PROJECT/'artifacts/extended_report';output.mkdir(exist_ok=True)
    for stage,script in stages:
        (output/'remaining_progress.json').write_text(json.dumps(dict(stage=stage,status='running'),indent=2),encoding='utf-8')
        print('REMAINING PIPELINE:',stage,flush=True)
        subprocess.run([sys.executable,'-u',str(PROJECT/script)],cwd=PROJECT,check=True)
    (output/'remaining_progress.json').write_text(json.dumps(dict(stage='complete',status='complete'),indent=2),encoding='utf-8')
    print('REMAINING EXPERIMENTS AND OVERALL BASELINE COMPLETE',flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        target=PROJECT/'artifacts/extended_report';target.mkdir(exist_ok=True)
        (target/'remaining_failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
