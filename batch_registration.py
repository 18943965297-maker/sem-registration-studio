"""Explicit pairing, isolated batch outputs, and no-refit registration audit."""
import csv
import os
import datetime as dt
import json
import re
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image
from registration_pipeline import (read_sem_metadata, infer_magnification, read_image_unicode,
    register_arrays, save_registration_result, _detect_and_match, _ncc, _grid_coverage,
    normalize_registration_image, write_png_unicode)

DEFAULT_OUTPUT = Path(os.environ.get('SEM_WORKSPACE', Path(__file__).resolve().parent / 'workspace')).expanduser().resolve() / 'batch_runs'
EXTENSIONS = {'.tif','.tiff','.png'}


def dump(path, data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def new_run(output, prefix):
    folder = Path(output)/f'{prefix}_{dt.datetime.now():%Y%m%d_%H%M%S_%f}'
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def image_key(relative):
    parts = list(Path(relative).with_suffix('').parts)
    if len(parts)==1 and '__' in parts[0]:
        parts=parts[0].split('__')
    # Dataset conditions are attached to the specimen directory, not the field ID.
    if len(parts)>1:
        specimen=re.match(r'^(\d+)(?:\D|$)',parts[0])
        if specimen:
            parts[0]=specimen[1]
    return '/'.join(parts).casefold()


def find_sides(root):
    root=Path(root)
    def get(side):
        folders=[p for p in root.iterdir() if p.is_dir() and p.name.casefold() in
                 {side,f'sem（{side}）',f'sem({side})'}]
        if len(folders)!=1:
            raise ValueError(f'需要唯一的{side}或SEM（{side}）子文件夹；不能猜测输入分组')
        return folders[0]
    return get('before'),get('after')


def scan_dataset(root):
    root=Path(root).resolve()
    sides=find_sides(root)
    calibration={}
    table=root/'pixel_calibration.csv'
    if table.exists():
        with table.open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream):
                key=(row.get('group','').lower(),row.get('filename','').casefold())
                if key in calibration:
                    raise ValueError(f'标定表有重复键：{key}')
                calibration[key]=row
    inventory={'before':[],'after':[]}
    excluded=[]
    for side,folder in zip(inventory,sides):
        for path in sorted(folder.rglob('*')):
            if path.suffix.lower() not in EXTENSIONS or not path.is_file():
                continue
            relative=path.relative_to(folder)
            if any(s in str(relative).casefold() for s in ('eds','mapping','总图','map data')):
                excluded.append({'path':str(path),'reason':'EDS/元素图/总图不自动参加配准'})
                continue
            record={'path':str(path),'relative':str(relative),'key':image_key(relative),'error':''}
            try:
                with Image.open(path) as im:
                    width,height=im.size
                    mode=im.mode
                meta=read_sem_metadata(path)
                record.update(width=width,height=height,mode=mode,
                              bit_depth=16 if mode in {'I;16','I;16B','I;16L'} else (8 if mode in {'L','RGB','RGBA'} else None),
                              magnification=meta.get('magnification') or infer_magnification(path),
                              pixel_size_um=meta.get('pixel_size_x_um'),pixel_y_um=meta.get('pixel_size_y_um'),
                              scale_source=meta.get('metadata_source'))
                row=calibration.get((side,path.name.casefold()))
                if row and not record['pixel_size_um']:
                    if (int(row['analysis_width_px']),int(row['analysis_height_px']))!=(width,height):
                        raise ValueError('标定表尺寸与文件不一致')
                    record.update(pixel_size_um=float(row['pixel_width_um']),pixel_y_um=float(row['pixel_height_um']),
                                  scale_source='pixel_calibration.csv（需确认标定可靠性）')
            except Exception as exc:
                record['error']=str(exc)
            inventory[side].append(record)
    def natural(path):
        return [int(x) if x.isdigit() else x.casefold() for x in re.split(r'(\d+)', Path(path).name)]

    def group_key(item):
        rel=Path(item['relative'])
        parts=list(rel.with_suffix('').parts)
        if len(parts)==1 and '__' in parts[0]:
            parts=parts[0].split('__')
        condition=re.match(r'^(\d+)', parts[0]).group(1) if parts and re.match(r'^(\d+)',parts[0]) else parts[0]
        leaf=parts[-2] if len(parts)>=3 else (parts[1] if len(parts)>=2 else '')
        # Before/After may spell the same leaf as "3号-未浸泡B" vs "3B".
        # Keep the stable alpha-numeric identity and remove the condition prefix.
        leaf=re.sub(r'[^a-zA-Z0-9]+','',leaf).casefold()
        if condition and leaf.startswith(condition.casefold()):
            leaf=leaf[len(condition):]
        mag=item.get('magnification')
        if mag:
            scale=('mag', round(float(mag)/50)*50)
        elif item.get('pixel_size_um'):
            # The no-bottom dataset has no HDR magnification, but its calibration
            # table gives stable physical sampling groups.
            # Exact calibrated values differ between the two microscopes/data
            # exports. For no-HDR folders, match by physical-scale rank inside
            # the same condition/field rather than requiring equal numbers.
            scale=('nomag', '')
        else:
            scale=('unknown', '')
        return condition, leaf, scale

    grouped=defaultdict(lambda: {'before':[],'after':[]})
    for side,items in inventory.items():
        for item in items:
            grouped[group_key(item)][side].append(item)
    pairs=[]
    for key, sides in sorted(grouped.items(), key=lambda kv: str(kv[0])):
        def order(item):
            return (-(float(item.get('pixel_size_um')) if item.get('pixel_size_um') else 0.0), natural(item['relative']))
        before_items=sorted(sides['before'],key=order)
        after_items=sorted(sides['after'],key=order)
        # A group with more than one image on either side is genuinely
        # ambiguous (e.g. extra pre-reaction magnifications or duplicate
        # exports).  Never silently zip these by filename/order.  Keep an
        # explicit empty review row so the UI can show the candidates and the
        # operator can choose a pair by looking at the images.
        if len(before_items) == 1 and len(after_items) == 1:
            pairs.append({'before':before_items[0]['path'],'after':after_items[0]['path'],'confirmed':False,
                          'reason':f'条件/视野/尺度组内唯一候选：{key}; 需看图确认'})
        elif before_items or after_items:
            pairs.append({'before':'','after':'','confirmed':False,
                          'candidates_before':[item['path'] for item in before_items],
                          'candidates_after':[item['path'] for item in after_items],
                          'reason':f'条件/视野/尺度组有多个候选，禁止自动配对：{key}'})
    return {'root':str(root),'inventory':inventory,'excluded':excluded,'pairs':pairs,
            'cropped_suggestion':table.exists() or 'no bottom' in root.name.casefold()}


def validate_scales(before,after):
    for item in (before,after):
        if item.get('error'):
            raise ValueError(item['error'])
        for name in ('pixel_size_um','pixel_y_um','magnification'):
            value=item.get(name)
            if value is not None and (not np.isfinite(value) or value<=0):
                raise ValueError('倍率/像素尺寸必须为有限正数')
        px,py=item.get('pixel_size_um'),item.get('pixel_y_um')
        if px and py and abs(px/py-1)>.02:
            raise ValueError('非方形物理像素，当前等比例尺度处理不适用，需人工复核')
    mags=[p.get('magnification') for p in (before,after)]
    pixels=[p.get('pixel_size_um') for p in (before,after)]
    if all(mags) and max(mags)/min(mags)>1.08:
        raise ValueError('倍率层级不同，批量变化配准不自动跨倍率')
    if not all(mags) and not all(pixels):
        raise ValueError('两侧倍率或物理像素标定不完整，不能猜测尺度')
    if not all(mags) and max(pixels)/min(pixels)>1.25:
        raise ValueError('倍率未知且像素尺度差较大，需要人工确认，不自动跨倍率')


def run_batch(scan, output, crop_footer, matcher='classic', progress=lambda msg:None, stopped=lambda:False):
    folder=new_run(output,'batch')
    dump(folder/'pairing_plan.json',scan)
    lookup={p['path']:p for records in scan['inventory'].values() for p in records}
    records=[]
    for index,pair in enumerate(scan['pairs']):
        if stopped():
            break
        record={'index':index+1,**pair,'status':'skipped'}
        progress(f"{index+1}/{len(scan['pairs'])}：{Path(pair['after']).name}")
        try:
            if not pair.get('confirmed') or not pair.get('before'):
                record['reason']='未确认配对，跳过'
            else:
                b,a=lookup[pair['before']],lookup[pair['after']]
                validate_scales(b,a)
                result=register_arrays(read_image_unicode(b['path']),read_image_unicode(a['path']),
                    b.get('magnification'),a.get('magnification'),b.get('pixel_size_um'),a.get('pixel_size_um'),
                    matcher=matcher,crop_footer=crop_footer)
                group='accepted' if result.metrics.quality in ('优秀','良好') else 'review'
                dest=folder/group/f'pair_{index+1:05d}'
                files=save_registration_result(result,dest)
                row={'pair_id':dest.name,'before':files['before'],'after':files['after'],'mask':files['mask'],
                     'diff':files['diff'],'registration_metrics':files['metrics'],'annotation_dir':str(dest)}
                dump(dest/'source_pair.json',{'before':b,'after':a,'crop_footer':crop_footer,'matcher':matcher})
                # Only accepted pairs are immediately reopenable as annotation tasks.
                if group=='accepted':
                    dump(dest/'annotation_project.json',{'version':1,'row':row,'mode':'双侧孔隙'})
                record.update(status=group,quality=result.metrics.quality,folder=str(dest),
                              median_px=result.metrics.reprojection_median_px if np.isfinite(result.metrics.reprojection_median_px) else None,
                              p95_px=result.metrics.reprojection_p95_px if np.isfinite(result.metrics.reprojection_p95_px) else None,
                              reason=result.metrics.warning)
        except Exception as exc:
            record.update(status='failed',reason=str(exc))
        records.append(record)
        dump(folder/'batch_summary.json',{'records':records,'completed':False})
    dump(folder/'batch_summary.json',{'records':records,'completed':len(records)==len(scan['pairs']),
                                     'unprocessed':len(scan['pairs'])-len(records)})
    return folder,records


def audit_arrays(before,after):
    """Measure current coordinate displacement. No fitted transform or image warp."""
    if before.shape!=after.shape:
        raise ValueError('图像尺寸不同；检测不会替你缩放或重新配准')
    if np.array_equal(before,after):
        return {'status':'需复核','reason':'两张图完全相同，需排除误选同图','ncc':1.0}
    _,kr,km,good=_detect_and_match(before,after)
    # Mutual one-to-one descriptor matches reduce ambiguous repeated textures.
    _,reverse_ref,reverse_mov,reverse_good=_detect_and_match(after,before)
    reverse={(tuple(np.round(reverse_mov[m.queryIdx].pt,3)),tuple(np.round(reverse_ref[m.trainIdx].pt,3))) for m in reverse_good}
    pairs=[(km[m.queryIdx].pt,kr[m.trainIdx].pt) for m in good
           if (tuple(np.round(kr[m.trainIdx].pt,3)),tuple(np.round(km[m.queryIdx].pt,3))) in reverse]
    if len(pairs)<6:
        return {'status':'证据不足','reason':'双向一致匹配不足6对，不能判定合格','matches':len(pairs)}
    points=np.asarray(pairs)
    errors=np.linalg.norm(points[:,0]-points[:,1],axis=1)
    median,p95=float(np.median(errors)),float(np.percentile(errors,95))
    coverage=_grid_coverage(points[:,1],before.shape)
    ncc=_ncc(before,after,np.ones(before.shape,bool))
    tiles=[]
    h,w=before.shape
    for y in range(4):
        for x in range(4):
            a=before[y*h//4:(y+1)*h//4,x*w//4:(x+1)*w//4]
            b=after[y*h//4:(y+1)*h//4,x*w//4:(x+1)*w//4]
            tiles.append({'row':y,'column':x,'ncc':_ncc(a,b,np.ones(a.shape,bool))})
    okay=len(pairs)>=10 and median<=2 and p95<=5 and coverage>=.125 and ncc>=.15
    return {'status':'自动初检通过（仍需人工复核）' if okay else '需复核','matches':len(pairs),
            'direct_displacement_median_px':median,'direct_displacement_p95_px':p95,
            'coverage':coverage,'ncc':ncc,'tiles_4x4':tiles,
            'matched_after_before_xy':points.tolist(),'direct_displacements_px':errors.tolist(),
            'reason':'现有坐标直接位移；未拟合扣除新变换。匹配点不是人工独立真值，真实变化也可能降低相关性。'}


def audit_folder(root,output,progress=lambda msg:None,stopped=lambda:False):
    root=Path(root).resolve()
    pairs=[]
    from dataset_paths import resolve_row
    for project in sorted(root.rglob('annotation_project.json')):
        saved = json.loads(project.read_text(encoding='utf-8'))
        row = resolve_row(saved['row'], project.parent)
        b, a = Path(row['before']), Path(row['after'])
        if b.exists() and a.exists():
            pairs.append((b,a))
    for before in sorted(root.rglob('before_core.png')):
        after=before.parent/'after_registered_core.png'
        if after.exists() and (before,after) not in pairs:
            pairs.append((before,after))
    if not pairs:
        b,a=find_sides(root)
        lookup=defaultdict(list)
        for path in b.rglob('*'):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                lookup[path.relative_to(b).with_suffix('').as_posix()].append(path)
        used=set()
        for path in sorted(a.rglob('*')):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                key=path.relative_to(a).with_suffix('').as_posix()
                candidates=lookup.get(key,[])
                pairs.append((candidates[0] if len(candidates)==1 else None,path))
                used.add(key)
        for key,items in lookup.items():
            if key not in used:
                pairs.extend((p,None) for p in items)
    if not pairs:
        raise ValueError('没有找到配准图对：需要before_core/after_registered_core，或before/after下同名对应图')
    folder=new_run(output,'audit')
    records=[]
    for index,(b,a) in enumerate(pairs):
        if stopped():
            break
        progress(f'质量检测 {index+1}/{len(pairs)}：{a.name if a else b.name}')
        report={'before':str(b) if b else None,'after':str(a) if a else None}
        try:
            if b is None or a is None:
                raise ValueError('无唯一同名Before/After，跳过，不能猜测配对')
            bi,ai=read_image_unicode(b),read_image_unicode(a)
            report.update(audit_arrays(bi,ai))
            yy,xx=np.indices(bi.shape)
            board=normalize_registration_image(bi)
            other=normalize_registration_image(ai)
            choose=(xx//64+yy//64)%2==1
            board[choose]=other[choose]
            display_name = b.stem if b.name != 'before_core.png' else b.parent.name
            path=folder/f'{display_name}_配准质量检查.png'
            if path.exists():
                import hashlib
                suffix=hashlib.sha256(str(b).encode('utf-8')).hexdigest()[:8]
                path=folder/f'{display_name}_{suffix}_配准质量检查.png'
            write_png_unicode(path,board)
            report['checkerboard']=str(path)
        except Exception as exc:
            report.update(status='无法检测',reason=str(exc))
        records.append(report)
        dump(folder/'audit_summary.json',{'records':records,'completed':False})
    dump(folder/'audit_summary.json',{'records':records,'completed':len(records)==len(pairs),
                                     'unprocessed':len(pairs)-len(records)})
    return folder,records
