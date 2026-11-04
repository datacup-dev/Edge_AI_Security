"""Download and verify the fixed checkpoint training-data GitHub release.

Run on the lab server from Coding/edge_ids. Extract as described in RESTORE.md.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request

REPO='datacup-dev/Edge_AI_Security'
TAG='training-snapshot-20261104-121037'

def checksum(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return 'sha256:'+h.hexdigest()

def main(destination):
    destination.mkdir(parents=True,exist_ok=True)
    headers={'User-Agent':'edge-ids-data-download','Accept':'application/vnd.github+json'}
    request=urllib.request.Request(f'https://api.github.com/repos/{REPO}/releases/tags/{TAG}',headers=headers)
    with urllib.request.urlopen(request,timeout=90) as response:release=json.load(response)
    assert not release['draft']
    selected=[a for a in release['assets'] if re.fullmatch(r'training-data\.tar\.gz\.part\d{3}',a['name']) or a['name'] in ['SHA256SUMS','RESTORE.md','migration_inventory.json','package.json']]
    assert any(a['name']=='package.json' for a in selected),'Release missing package metadata'
    for asset in sorted(selected,key=lambda a:a['name']):
        target=destination/asset['name'];expected=asset.get('digest')
        assert expected and expected.startswith('sha256:'),'Missing published content checksum'
        if target.exists() and target.stat().st_size==asset['size'] and checksum(target)==expected:
            print('VERIFIED',target.name,flush=True);continue
        partial=target.with_name(target.name+'.download')
        for attempt in range(4):
            try:
                offset=partial.stat().st_size if partial.exists() else 0
                request=urllib.request.Request(asset['browser_download_url'],headers={'User-Agent':'edge-ids-data-download',**({'Range':f'bytes={offset}-'} if offset else {})})
                with urllib.request.urlopen(request,timeout=180) as response:
                    if offset and response.status==206:
                        assert response.headers['Content-Range'].startswith(f'bytes {offset}-')
                        mode='ab'
                    else:mode='wb'
                    with partial.open(mode) as stream:
                        last=time.monotonic()
                        while block:=response.read(2*1024*1024):
                            stream.write(block)
                            if time.monotonic()-last>20:
                                print('DOWNLOAD',target.name,stream.tell(),'/',asset['size'],flush=True);last=time.monotonic()
                assert partial.stat().st_size==asset['size'] and checksum(partial)==expected,'Downloaded checksum mismatch'
                partial.replace(target);print('VERIFIED',target.name,flush=True);break
            except OSError:
                if attempt==3:raise
                time.sleep(5)
    package=json.loads((destination/'package.json').read_text())
    for part in package['parts']:
        p=destination/part['name'];assert p.stat().st_size==part['bytes'] and checksum(p)=='sha256:'+part['sha256']
    print('All training-data assets downloaded and verified. Follow',destination/'RESTORE.md')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--destination',type=Path,default=Path('training-data-download'));args=parser.parse_args();main(args.destination)
