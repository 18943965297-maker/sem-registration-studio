"""Validate named dataset files plus an isolated editor save/reopen smoke test."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile
import tkinter as tk
from unittest.mock import patch

import numpy as np
from PIL import Image

from build_named_sem_dataset import (OUTPUT, load, dump, sha, register_arrays, raw_image,
    export_pair, names)
from dataset_paths import resolve_row, portable_row
from annotation_modes import AnnotationLayers, MODES
from mask_editor_app import MaskEditorApp


def editor_roundtrip(row, base):
    # GUI and SAM routing are exercised on temporary files, never on user masks.
    with tempfile.TemporaryDirectory() as temp:
        folder=Path(temp);copy=dict(resolve_row(row,base))
        for key in ('before','after','mask','diff'):
            dest=folder/Path(copy[key]).name
            shutil.copy2(copy[key],dest);copy[key]=str(dest)
        copy['annotation_dir']=str(folder)
        root=tk.Tk();root.withdraw()
        try:
            with patch.object(MaskEditorApp,'check_sam_service'),patch('mask_editor_app.messagebox.showinfo'):
                app=MaskEditorApp(root);app.rows=[copy];app.idx=0;app.load_current()
                shape=app.mask.shape;mask=np.zeros(shape,bool);mask[8:12,8:12]=True
                for side in ('before','after'):
                    app.sam_candidates[side]=mask.copy()
                    with patch.object(app,'_selected_sam_side',return_value=side):app.accept_sam_candidate()
                assert app.annotations.layers['before']['mask'].any()
                assert app.annotations.layers['after']['mask'].any()
                app.undo();assert not app.mask.any();app.redo();assert app.mask.any()
                app.save_mask()
                saved=load(folder/'annotation_project.json')
                assert not Path(saved['row']['before']).is_absolute()
                app.rows=[resolve_row(saved['row'],folder)];app.idx=0;app.load_current()
                assert app.annotations.layers['before']['mask'][9,9]
                assert app.annotations.layers['after']['mask'][9,9]
                for mode,key,value in ((MODES[1],'binary',1),(MODES[2],'multi',255)):
                    app.annotation_mode.set(mode);app.change_annotation_mode()
                    app.mask[15,15]=value;app.unsaved=True;app.save_mask()
                labels=AnnotationLayers(shape,folder,filename_prefix=copy['label_prefix'])
                assert labels.layers['binary']['mask'][15,15]
                assert labels.layers['multi']['mask'][15,15]==255
                for p in folder.rglob('*.png'):
                    assert '组' in p.name and '倍' in p.name,p
                assert app.before_pane.base_pil.mode=='RGB'
        finally:root.destroy()
    return True


def sample():
    inv=load(OUTPUT/'source_inventory.json')
    # Real same-field pairs span 500x/2000x and the difficult 20000x example.
    wanted=[('1/1-1/2.tif','1-CO2-60MPa-1m/1-1/1.tif'),
            ('1/1-1/3.tif','1-CO2-60MPa-1m/1-1/2.tif'),
            ('1/1-4/13.tif','1-CO2-60MPa-1m/1-4/3.tif')]
    reports=[]
    for before,after in wanted:
        b=next(r for r in inv['records'] if r['side']=='before' and r['original_relative_path']==before)
        a=next(r for r in inv['records'] if r['side']=='after' and r['original_relative_path']==after)
        result=register_arrays(raw_image(b['path']),raw_image(a['path']),
            b['magnification'],a['magnification'],b['pixel_size_um'],a['pixel_size_um'],
            before_content_bounds=b['content_bounds'],after_content_bounds=a['content_bounds'])
        with tempfile.TemporaryDirectory() as d:
            row=export_pair(Path(d),b,a,result,{'strong':False,'note':'isolated smoke'},False)['row']
            row['before_raw']=str(OUTPUT/b['archive_relative_path']);row['after_raw']=str(OUTPUT/a['archive_relative_path'])
            editor_roundtrip(row,Path(d))
        assert sha(b['path'])==b['source_sha256'] and sha(a['path'])==a['source_sha256']
        reports.append({'before':before,'after':after,'quality':result.metrics.quality,
            'median_px':result.metrics.reprojection_median_px,'ecc':result.ecc_validation,
            'ui_save_reopen':True})
    dump(OUTPUT/'sample_validation.json',reports)
    print(json.dumps(reports,ensure_ascii=False),flush=True)


def verify(output):
    summary=load(output/'batch_summary.json');inv=load(output/'source_inventory.json')
    assert summary['completed']
    dtype=Counter();paired=[];ready_before=set();good_rows=[]
    for rec in summary['records']:
        if not rec.get('row'):continue
        row=resolve_row(rec['row'],output);folder=Path(row['annotation_dir'])
        with Image.open(row['before']) as im:b=np.array(im)
        with Image.open(row['after']) as im:a=np.array(im)
        with Image.open(row['mask']) as im:blank=np.array(im)
        assert b.shape==a.shape==blank.shape
        assert not blank.any()
        source=load(folder/'source_metadata.json')
        for side,array in (('before',b),('after',a)):
            expected='uint16' if source[side]['metadata']['mode'].startswith('I;16') else 'uint8'
            assert str(array.dtype)==expected
            dtype[(side,str(array.dtype))]+=1
        for p in folder.glob('*.png'):assert '组' in p.name and '倍' in p.name,p
        assert Path(row['before_raw']).exists() and Path(row['after_raw']).exists()
        saved=load(folder/'annotation_project.json')
        assert resolve_row(saved['row'],folder)['before']==row['before']
        assert saved['row']['annotation_status']=='not_started'
        transforms=load(folder/'transforms.json')
        for side in ('before','after'):
            np.testing.assert_allclose(np.array(transforms[f'label_to_{side}_raw'])@np.array(transforms[f'{side}_raw_to_label']),np.eye(3),atol=1e-7)
        if rec['status']=='ready_to_annotate':
            assert rec['before'] not in ready_before
            ready_before.add(rec['before']);good_rows.append(rec['row'])
        paired.append(row['pair_id'])
    # Hash raw files again after the complete run, and check archived copies.
    for r in inv['records']:
        assert sha(r['path'])==r['source_sha256']
        assert sha(output/r['archive_relative_path'])==r['source_sha256']
        if r.get('hdr_sha256'):
            assert sha(r['metadata']['header_path'])==r['hdr_sha256']
            assert sha(output/r['hdr_archive_relative_path'])==r['hdr_sha256']
    if good_rows:editor_roundtrip(good_rows[0],output)
    report={'validated_pairs':len(paired),'raw_tiffs_hashed':len(inv['records']),
            'raw_tiffs_unchanged':True,'archive_verified':True,'ready_pairs':len(ready_before),
            'dtypes':{str(k):v for k,v in dtype.items()},'named_mask_save_reload':bool(good_rows),
            'blank_masks_are_unannotated':True}
    dump(output/'validation_report.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sample',action='store_true');parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    if args.sample:sample()
    else:verify(args.output)
