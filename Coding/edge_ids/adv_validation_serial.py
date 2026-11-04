"""Memory-limited resumable validation scheduler: one lane at a time."""
import gc
import time
import adv_validation_control
from adv_validation_control import engine

engine.TEST.joinpath('failure_history').mkdir(exist_ok=True)
for p in (engine.TEST/'workers').glob('*failure.json'):
    p.rename(engine.TEST/'failure_history'/f'{time.time_ns()}_{p.name}')
pending={(lane,name) for lane in engine.LANES for name in engine.NAMES}
while pending:
    for lane,name in list(sorted(pending)):
        try:
            if engine.generate(lane,test_dataset=name,one_cycle=True,heldout_role='validation'):
                pending.remove((lane,name))
        except (OSError,MemoryError) as error:
            if not isinstance(error,MemoryError) and getattr(error,'winerror',None)!=1455:raise
            print('Memory pressure: retained cursor; retrying',flush=True)
            time.sleep(30)
        finally:gc.collect()
    engine.report()
engine.audit()
