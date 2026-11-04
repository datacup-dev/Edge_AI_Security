"""Validation-only attack controller sharing the already checked held-out engine."""
import argparse
import traceback
import adv_test_control as engine
from adv_mixed_data import PROJECT,save

engine.TEST=PROJECT/'artifacts/separate_validation_attacks'
engine.ROLE='validation'
engine.SPLIT_CODE=1
engine.ENTRY='adv_validation_control.py'
engine.REPORT_FILE='VALIDATION_ATTACK_REPORT.md'
engine.PROTOCOL_FILE='VALIDATION_ATTACKS.md'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','launch','worker','report','audit']);parser.add_argument('--lane',choices=engine.LANES);parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    try:
        if args.action=='worker':engine.worker(args.lane)
        elif args.action=='audit':engine.audit(args.partial)
        else:{'prepare':engine.prepare,'launch':engine.launch,'report':engine.report}[args.action]()
    except Exception:
        if args.action=='worker':save(engine.TEST/'workers'/f'{args.lane}_failure.json',dict(traceback=traceback.format_exc()))
        raise
