"""Build a source-preserving, explicitly named SEM annotation dataset.

Every output image carries condition/group/magnification and source identity.
Matching is restricted to compatible specimen/scale groups, with ambiguity
recorded separately. No annotation truth or human approval is synthesized.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from functools import lru_cache
import csv
import datetime as dt
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.parse import quote

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from dataset_paths import portable_row, resolve_row
from registration_pipeline import (
    read_sem_metadata, find_sem_header, read_image_unicode, normalize_registration_image,
    register_arrays, save_registration_result, _grid_coverage,
)

CODE_ROOT = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get('SEM_WORKSPACE', CODE_ROOT / 'workspace')).expanduser().resolve()
SOURCE = WORKSPACE / 'input'
OUTPUT = WORKSPACE / 'named_dataset'
VERSION = 'named-full-tiff-v1'


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [plain(v) for v in value]
    if isinstance(value, bytes):
        return {'encoding':'base64', 'data':base64.b64encode(value).decode('ascii')}
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str,int,bool)):
        return value
    return str(value)


def dump(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # Replace only this builder's own progress JSON, not image/source files.
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(plain(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def safe(text):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]','-',str(text)).rstrip('. ')


def number(value):
    return f'{value:g}' if isinstance(value,(int,float)) else '未知'


def nominal_mag(mag):
    if not mag or not np.isfinite(mag):
        return None
    common=np.array([20,30,50,100,200,500,1000,2000,5000,10000,20000,50000,100000],float)
    choice=common[np.argmin(abs(common/mag-1))]
    return int(choice) if abs(choice/mag-1)<.025 else round(float(mag),2)


def acquisition(path):
    with Image.open(path) as im:
        tags=dict(im.tag_v2)
        info={'shape_hw':[im.height,im.width], 'mode':im.mode, 'frames':getattr(im,'n_frames',1),
              'bits_per_sample':plain(tags.get(258)), 'orientation':tags.get(274,1),
              'tiff_tags':plain(tags), 'pillow_info':plain(im.info)}
        texts=[]
        for tag in (50431,270):
            v=tags.get(tag,b'')
            if isinstance(v,bytes):texts.append(v.decode('utf-8',errors='replace'))
            else:texts.append(str(v))
    hdr=find_sem_header(path)
    if hdr:
        texts.append(hdr.read_bytes().decode('utf-8-sig',errors='replace'))
    def values(key):
        return re.findall(r'(?:^|[\r\n\x00])'+re.escape(key)+r'\s*=\s*([^\r\n\x00]+)', '\n'.join(texts))
    sem=read_sem_metadata(path)
    strips=[]
    for s in values('ImageStripSize'):
        try:strips.append(int(s))
        except ValueError:pass
    info.update(sem_metadata=sem, header_path=str(hdr) if hdr else None,
                detectors=list(dict.fromkeys(values('Detector'))),
                acquisition_dates=list(dict.fromkeys(values('Date'))),
                image_strip_values=list(dict.fromkeys(strips)))
    info['metadata_conflicts']=[]
    for key in ('Magnification','PixelSizeX','PixelSizeY'):
        nums=[]
        for s in values(key):
            try:nums.append(float(s))
            except ValueError:pass
        if nums and min(nums)>0 and max(nums)/min(nums)>1.02:
            info['metadata_conflicts'].append({'field':key,'values':nums})
    return info


def content_bounds(info):
    h,w=info['shape_hw'];strip=info['image_strip_values']
    if info['frames']!=1:
        raise ValueError('多页TIFF，需确认使用哪一页')
    if info['orientation']!=1:
        raise ValueError('TIFF方向不为1，需要核查坐标方向')
    if len(strip)!=1 or not 0<strip[0]<h-64:
        raise ValueError('无唯一有效ImageStripSize，不猜测裁栏高度')
    if info['metadata_conflicts']:
        raise ValueError('嵌入/HDR尺度元数据冲突')
    return [0,0,w,h-strip[0]]


def specimen_key(group, subdir):
    if group in ('1','2'):
        found=re.search(r'(?:^|\.)([12]-\d+)',subdir)
        return found[1] if found else subdir.split('（')[0]
    if group=='3':
        first=subdir.split('/')[0]
        found=re.search(r'([ABab])$',first)
        return found[1].upper() if found else first
    return subdir


def prepare(source, output):
    if (output/'source_inventory.json').exists():
        inv=load(output/'source_inventory.json')
        if inv['source']!=str(source.resolve()):raise ValueError('输出目录的来源不同')
        return inv
    output.mkdir(parents=True,exist_ok=True)
    after_root=source/'SEM（after）'
    conditions={p.name.split('-',1)[0]:p.name for p in after_root.iterdir() if p.is_dir()}
    records=[];excluded=[]
    for side in ('before','after'):
        root=source/f'SEM（{side}）'
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in ('.tif','.tiff'):continue
            rel=path.relative_to(root)
            if any(term in str(rel).casefold() for term in ('eds','mapping','总图','map data')):
                excluded.append({'path':str(path),'reason':'EDS/元素图/总图，未纳入孔隙图像配准'})
                continue
            group=rel.parts[0].split('-',1)[0];sub='/'.join(rel.parts[1:-1])
            info=acquisition(path)
            rec={'side':side,'group':group,'condition':conditions[group], 'subdirectory':sub,
                 'specimen_key':specimen_key(group,sub), 'original_filename':path.name,
                 'stem':path.stem,'path':str(path), 'original_relative_path':rel.as_posix(),
                 'source_sha256':sha(path),'metadata':info, 'error':''}
            sem=info['sem_metadata'];rec.update(magnification=sem['magnification'],
                pixel_size_um=sem['pixel_size_x_um'],pixel_y_um=sem['pixel_size_y_um'],
                nominal_magnification=nominal_mag(sem['magnification']))
            try:
                rec['content_bounds']=content_bounds(info)
                if not rec['nominal_magnification'] or not rec['pixel_size_um'] or not rec['pixel_y_um']:
                    raise ValueError('倍率/物理像素缺失')
                if abs(rec['pixel_size_um']/rec['pixel_y_um']-1)>.02:
                    raise ValueError('非方形物理像素，当前保守模型不自动处理')
            except ValueError as exc:rec['error']=str(exc)
            archive=output/'sources'/side/rel
            archive.parent.mkdir(parents=True,exist_ok=True)
            if not archive.exists():shutil.copy2(path,archive)
            if sha(archive)!=rec['source_sha256']:raise ValueError(f'归档校验失败：{archive}')
            rec['archive_relative_path']=archive.relative_to(output).as_posix()
            if info['header_path']:
                hdr=Path(info['header_path']);dest=archive.parent/hdr.name
                if not dest.exists():shutil.copy2(hdr,dest)
                if sha(hdr)!=sha(dest):raise ValueError(f'HDR归档校验失败：{hdr}')
                rec['hdr_archive_relative_path']=dest.relative_to(output).as_posix()
                rec['hdr_sha256']=sha(hdr)
            records.append(rec)
    inv={'source':str(source.resolve()),'version':VERSION,'records':records,'excluded':excluded,
         'conditions':conditions,'counts':dict(Counter(r['side'] for r in records))}
    dump(output/'source_inventory.json',inv)
    print('INVENTORY',inv['counts'],'errors',dict(Counter(r['error'] for r in records)),flush=True)
    return inv


@lru_cache(maxsize=12)
def raw_image(path):
    return read_image_unicode(path)


def cropped(record):
    x0,y0,x1,y1=record['content_bounds']
    return raw_image(record['path'])[y0:y1,x0:x1]


def compatible(b,a):
    return not (b['error'] or a['error']) and (
        b['group']==a['group'] and b['specimen_key']==a['specimen_key']
        and b['nominal_magnification']==a['nominal_magnification']
        and max(b['magnification'],a['magnification'])/min(b['magnification'],a['magnification'])<1.08)


def screen_correspondences(b,a,bpoints,apoints):
    """Fit a split of candidate correspondences; report withheld evidence separately."""
    n=len(bpoints)
    if n<10:return None
    target=max(b['pixel_size_um'],a['pixel_size_um'])
    bf,af=b['pixel_size_um']/target,a['pixel_size_um']/target
    dst=np.asarray(bpoints,np.float32)*bf+(bf-1)/2
    src=np.asarray(apoints,np.float32)*af+(af-1)/2
    held=np.arange(n)%5==0
    cv2.setRNGSeed(20260913)
    matrix,flags=cv2.estimateAffinePartial2D(src[~held],dst[~held],method=cv2.RANSAC,
        ransacReprojThreshold=3,maxIters=6000,confidence=.999)
    if matrix is None or flags is None:return None
    keep=flags.ravel().astype(bool);count=int(keep.sum());ratio=count/len(keep)
    scale=float(np.hypot(matrix[0,0],matrix[1,0]))
    errs=np.linalg.norm(cv2.transform(src[:,None],matrix)[:,0]-dst,axis=1)
    bounds=b['content_bounds'];shape=(round((bounds[3]-bounds[1])*bf),round((bounds[2]-bounds[0])*bf))
    coverage=_grid_coverage(dst[~held][keep],shape)
    held_good=int((errs[held]<=4).sum());held_total=int(held.sum())
    if count<6 or coverage<.125 or ratio<.10 or not .8<=scale<=1.25 or held_good<2:return None
    median=float(np.median(errs[~held][keep]))
    strong=count>=10 and ratio>=.25 and held_good/held_total>=.50
    score=count*math.sqrt(coverage)*(held_good/held_total)/(1+median)
    return {'score':score,'screen_inliers':count,'screen_inlier_ratio':ratio,'coverage':coverage,
            'withheld_consistent':held_good,'withheld_total':held_total,
            'screen_median_px':median,'screen_scale':scale,'strong':strong,
            'before_points_cropped':np.asarray(bpoints).tolist(),'after_points_cropped':np.asarray(apoints).tolist(),
            'withheld_indices':np.flatnonzero(held).tolist(),
            'note':'留出点只验证候选筛选；不是最终配准的独立人工真值'}


def screen(inv,output):
    checkpoint=output/'candidate_evidence.json'
    if checkpoint.exists() and load(checkpoint).get('completed'):return load(checkpoint)
    before=[r for r in inv['records'] if r['side']=='before']
    after=[r for r in inv['records'] if r['side']=='after']
    detector=cv2.SIFT_create(nfeatures=4000,contrastThreshold=.015,edgeThreshold=15)
    cache={};edges=[];issues=[];candidates_by_after={}
    def features(r):
        if r['path'] not in cache:
            kp,des=detector.detectAndCompute(normalize_registration_image(cropped(r)),None)
            cache[r['path']]=(np.float32([k.pt for k in kp]).reshape(-1,2),des)
        return cache[r['path']]
    for ai,a in enumerate(after):
        bs=[bi for bi,b in enumerate(before) if compatible(b,a)]
        candidates_by_after[str(ai)]=bs
        for bi in bs:
            b=before[bi]
            try:
                bp,bd=features(b);ap,ad=features(a)
                if bd is None or ad is None or len(bd)<2 or len(ad)<2:continue
                pairs=cv2.BFMatcher(cv2.NORM_L2).knnMatch(ad,bd,k=2)
                good=sorted([p[0] for p in pairs if len(p)==2 and p[0].distance<.74*p[1].distance],key=lambda m:m.distance)
                seen=set();good=[m for m in good if not(m.trainIdx in seen or seen.add(m.trainIdx))]
                e=screen_correspondences(b,a,[bp[m.trainIdx] for m in good],[ap[m.queryIdx] for m in good])
                if e:edges.append(dict(e,before_index=bi,after_index=ai,matcher='classic'))
            except Exception as exc:issues.append({'before':b['path'],'after':a['path'],'error':str(exc)})
        if (ai+1)%10==0 or ai+1==len(after):
            print(f'SCREEN {ai+1}/{len(after)} plausible={len(edges)}',flush=True)
    # Deep matching only for locally constrained, failed classical candidates.
    # Broad specimen groups without image evidence remain unmatched.
    for ai,a in enumerate(after):
        if any(e['after_index']==ai and e['strong'] for e in edges):continue
        bs=candidates_by_after[str(ai)]
        existing=sorted([e for e in edges if e['after_index']==ai],key=lambda e:e['score'],reverse=True)
        chosen=[e['before_index'] for e in existing[:2]] if existing else (bs if len(bs)<=3 else [])
        for bi in chosen:
            b=before[bi]
            try:
                from enhanced_matching import detect_lightglue
                _,bp,ap,matches=detect_lightglue(cropped(b),cropped(a))
                e=screen_correspondences(b,a,[bp[m.trainIdx].pt for m in matches],[ap[m.queryIdx].pt for m in matches])
                if e:
                    old=next((x for x in edges if x['before_index']==bi and x['after_index']==ai),None)
                    if old is None or (e['strong'],e['score'])>(old['strong'],old['score']):
                        if old:edges.remove(old)
                        edges.append(dict(e,before_index=bi,after_index=ai,matcher='lightglue'))
            except Exception as exc:issues.append({'stage':'lightglue','before':b['path'],'after':a['path'],'error':str(exc)})
        if chosen:print(f'FALLBACK {ai+1}/{len(after)} tried={len(chosen)} plausible={len(edges)}',flush=True)
    result={'completed':True,'edges':edges,'issues':issues,'candidates_by_after':candidates_by_after,
            'policy':'same condition/specimen/nominal scale; image features and withheld geometry; mutual best score margin 1.35'}
    dump(checkpoint,result)
    return result


def distinct(options):
    opts=sorted(options,key=lambda e:e['score'],reverse=True)
    if not opts:return None
    if len(opts)>1 and opts[0]['score']<opts[1]['score']*1.35:return None
    return opts[0]


def names(b,a):
    treatment=a['condition'].split('-',1)[1]
    base=f'{treatment}_第{a["group"]}组_{number(a["nominal_magnification"])}倍'
    if b['subdirectory']==a['subdirectory']:
        base+=f'_样品{a["subdirectory"] or "未单列"}'
    else:
        base+=f'_前样品{b["subdirectory"] or "未单列"}_后样品{a["subdirectory"] or "未单列"}'
    base=safe(base)
    pair=safe(f'{base}_前{b["stem"]}_后{a["stem"]}')
    uid=hashlib.sha256((b['original_relative_path']+'|'+a['original_relative_path']+'|'+b['source_sha256']+'|'+a['source_sha256']).encode()).hexdigest()
    files={'before':f'{pair}_Before.png','after':f'{pair}_After.png',
           'diff':f'{pair}_差异预览.png','mask':f'{pair}_空白初始Mask.png',
           'overlap':f'{pair}_有效重叠区.png','checkerboard':f'{pair}_棋盘格.png',
           'overlay':f'{pair}_叠加检查.png'}
    return pair,uid,files


def translate(x,y):return np.array([[1,0,x],[0,1,y],[0,0,1]],float)


def destination_folder(output,state,pair,b,a):
    folder=output/state/pair
    worst=folder/'change_multiclass'/(pair+'_变化多分类Mask.png')
    if len(str(worst))>=255:
        # Keep the full scientific name on every image. Only shorten a redundant
        # containing folder, preserving full source identity in the manifest.
        strip_note=lambda s:re.split(r'[（(]',s)[0]
        short=(f'{a["condition"].split("-",1)[1]}_第{a["group"]}组_{number(a["nominal_magnification"])}倍_'
               f'样品{a["subdirectory"] or "未单列"}_前{strip_note(b["stem"])}_后{strip_note(a["stem"])}')
        folder=output/state/safe(short)
    return folder


def matrices(result):
    bx,by,*_=result.before_crop_bounds_xyxy;ax,ay,*_=result.after_crop_bounds_xyxy
    cx,cy,*_=result.core_bounds_xyxy
    cb=translate(-bx,-by);ca=translate(-ax,-ay);core=translate(-cx,-cy)
    b=core@result.before_sampling@cb
    a=core@result.transform_after_to_before@ca
    return {'coordinate_order':'(x,y); integer pixel centres; half-open crop rectangles',
            'before_raw_to_label':b.tolist(),'after_raw_to_label':a.tolist(),
            'label_to_before_raw':np.linalg.inv(b).tolist(),'label_to_after_raw':np.linalg.inv(a).tolist(),
            'after_raw_to_before_raw':(np.linalg.inv(b)@a).tolist(),
            'before_crop_xyxy':result.before_crop_bounds_xyxy,'after_crop_xyxy':result.after_crop_bounds_xyxy,
            'core_crop_xyxy':result.core_bounds_xyxy,'target_pixel_size_um':result.metrics.target_pixel_size_um,
            'shape_hw':list(result.before_core.shape)}


def caption_preview(folder,row,b,a,result,files):
    w=1280;panel=604;font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    lines=[row['label_prefix'],f'实验条件：{a["condition"]}；自动状态：{row["registration_status"]}；尚未人工验收',
           f'Before：{b["original_filename"]} / {number(b["magnification"])}× / {b["metadata"]["mode"]}',
           f'After：{a["original_filename"]} / {number(a["magnification"])}× / {a["metadata"]["mode"]}',
           f'中位残差 {number(result.metrics.reprojection_median_px)} px；P95 {number(result.metrics.reprojection_p95_px)} px；内点 {result.metrics.inliers}']
    wrapped=[];probe=ImageDraw.Draw(Image.new('RGB',(1,1)))
    for line in lines:
        s=''
        for char in line:
            if probe.textlength(s+char,font=font)>w-40:wrapped.append(s);s=''
            s+=char
        wrapped.append(s)
    head=32*len(wrapped)+24
    canvas=Image.new('RGB',(w,head+2*(panel+40)+44),'#f4f6fa');draw=ImageDraw.Draw(canvas)
    draw.rectangle((0,0,w,head),fill='#172940')
    for i,line in enumerate(wrapped):draw.text((20,12+i*32),line,font=font,fill='white')
    ims=[normalize_registration_image(result.before_core),normalize_registration_image(result.after_core),
         read_image_unicode(files['checkerboard']),read_image_unicode(files['overlay'])]
    for i,(title,im) in enumerate(zip(('Before 反应前','After 反应后配准图','棋盘格','叠加检查'),ims)):
        x,y=20+(i%2)*636,head+(i//2)*(panel+40)
        draw.text((x,y+4),title,font=font,fill='#172940')
        pic=ImageOps.contain(Image.fromarray(im).convert('RGB'),(panel,panel),Image.Resampling.LANCZOS)
        canvas.paste(pic,(x+(panel-pic.width)//2,y+36+(panel-pic.height)//2))
    draw.text((20,canvas.height-36),'带标题图片仅供复核；标注使用独立Before/After PNG；空白Mask表示未标注。',font=font,fill='#31425b')
    out=folder/f'{row["label_prefix"]}_复核预览.jpg';canvas.save(out,quality=92)
    return out


def write_csv(path,rows,fields=None):
    fields=fields or sorted({k for row in rows for k in row}) or ['pair_id','before','after','mask','status']
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        writer.writerows(rows)


def export_pair(output,b,a,result,candidate,unambiguous):
    m=result.metrics
    warnings=[]
    passed=unambiguous and candidate['strong'] and m.quality in ('优秀','良好') and m.overlap_ratio>=.5 and .8<=m.transform_scale<=1.25
    if not unambiguous:warnings.append('候选不唯一或非双向最优，不能自动确认配对')
    if not candidate['strong']:warnings.append('图像对应证据不足自动通过门槛')
    if m.quality not in ('优秀','良好'):warnings.append(m.warning)
    if m.overlap_ratio<.5:warnings.append('有效重叠小于50%')
    if not .8<=m.transform_scale<=1.25:warnings.append('残余缩放超出保守范围')
    state='ready_to_annotate' if passed else 'needs_review'
    pair,uid,filenames=names(b,a)
    folder=destination_folder(output,state,pair,b,a)
    if folder.exists():raise ValueError(f'同名图对已存在：{folder}')
    files=save_registration_result(result,folder,filenames=filenames)
    row={'pair_id':pair,'pair_uid':uid,'label_prefix':pair,'after_condition':a['condition'],
         'leaf':a['subdirectory'] or '未单列','before_leaf':b['subdirectory'] or '未单列',
         'before_source_name':b['original_filename'],'after_source_name':a['original_filename'],
         'before':files['before'],'after':files['after'],'mask':files['mask'],'diff':files['diff'],
         'registration_metrics':files['metrics'],'annotation_dir':str(folder),
         'before_raw':str(output/b['archive_relative_path']),'after_raw':str(output/a['archive_relative_path']),
         'source_metadata':str(folder/'source_metadata.json'),
         'before_magnification':b['magnification'],'after_magnification':a['magnification'],
         'magnification_group':a['nominal_magnification'],'registration_status':state,
         'status':'ok' if passed else 'review','dataset_version':VERSION,
         'annotation_status':'not_started','human_review_status':'pending',
         'before_title':f'Before | {b["original_filename"]} | {number(b["magnification"])}×',
         'after_title':f'After | {a["original_filename"]} | {number(a["magnification"])}×'}
    dump(folder/'source_metadata.json',{'before':b,'after':a,'identity_basis':'原始目录名称，1m等时间标记未擅自释义'})
    dump(folder/'source_pair.json',{'before':b,'after':a,'pair_uid':uid,'human_confirmed':False,
         'pairing_status':'mutual_best_image_candidate' if unambiguous else 'ambiguous_candidate',
         'candidate':candidate,'review_reasons':warnings})
    dump(folder/'transforms.json',matrices(result))
    dump(folder/'annotation_project.json',{'version':1,'row':portable_row(row,folder),'mode':'双侧孔隙',
         'quality':'合格' if passed else '待复核','note':'自动筛选，尚未人工验收；空白Mask为初始化。'})
    preview=caption_preview(folder,row,b,a,result,files)
    return {'status':state,'folder':folder.relative_to(output).as_posix(),
            'row':portable_row(row,output),'preview':preview.relative_to(output).as_posix(),
            'before':b['path'],'after':a['path'],'reasons':warnings,
            'median_px':plain(m.reprojection_median_px),'p95_px':plain(m.reprojection_p95_px),
            'inliers':m.inliers,'coverage':m.coverage_ratio,'overlap':m.overlap_ratio}


def catalog(output,records):
    cards=[];esc=html.escape
    for r in records:
        if not r.get('row'):continue
        row=r['row'];q=lambda p:quote(p,safe='/')
        cards.append(f'<article data-status="{r["status"]}"><h2>{esc(row["pair_id"])}</h2>'
            f'<p>{esc(r["status"])} · 中位残差 {r["median_px"]} px · P95 {r["p95_px"]} px</p>'
            f'<a href="{q(r["preview"])}"><img loading="lazy" src="{q(r["preview"])}"></a>'
            f'<p><a href="{q(row["before"])}">Before</a>　<a href="{q(row["after"])}">After</a>　'
            f'<a href="{q(row["source_metadata"])}">原始信息</a></p></article>')
    page='''<!doctype html><meta charset="utf-8"><title>SEM 标注数据集</title>
<style>body{font:16px/1.6 "Microsoft YaHei",sans-serif;background:#edf2f7;color:#172940;margin:24px}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(540px,100%),1fr));gap:20px}
article{background:white;padding:18px;border-radius:10px}h2{font-size:18px;overflow-wrap:anywhere}img{width:100%}select{padding:8px}</style>
<h1>带底栏原始 TIFF · 配准与标注</h1><p>文件名含实验条件、组号、倍率及前后原图序号。主图和初始化Mask未写入标题；点击复核图放大检查。</p>
<p>ready_to_annotate = 自动筛选通过、尚未人工验收；needs_review = 配对或配准需要复核。空白Mask不代表没有变化。</p>
<select onchange="document.querySelectorAll('article').forEach(x=>x.hidden=this.value&&x.dataset.status!==this.value)">
<option value="">全部</option><option>ready_to_annotate</option><option>needs_review</option></select><main>'''+''.join(cards)+'</main>'
    (output/'打开数据集总览.html').write_text(page,encoding='utf-8')


def normalize_layout(output,records):
    """Shorten only redundant containing folders when Windows paths are too long."""
    output=Path(output).resolve();changes=[]
    for rec in records:
        if not rec.get('row'):continue
        old=(output/rec['folder']).resolve()
        source=load(old/'source_metadata.json')
        new=destination_folder(output,rec['status'],rec['row']['label_prefix'],source['before'],source['after']).resolve()
        if old==new:continue
        if not old.is_relative_to(output) or not new.is_relative_to(output) or old.parent!=new.parent:
            raise ValueError('目录调整超出当前数据集图对目录')
        if new.exists():raise ValueError(f'目标目录已经存在：{new}')
        before_hash={p.name:sha(p) for p in old.iterdir() if p.is_file()}
        old.rename(new)
        assert before_hash=={p.name:sha(p) for p in new.iterdir() if p.is_file()}
        oldrel=old.relative_to(output).as_posix();newrel=new.relative_to(output).as_posix()
        for key,value in list(rec['row'].items()):
            if isinstance(value,str) and (value==oldrel or value.startswith(oldrel+'/')):
                rec['row'][key]=newrel+value[len(oldrel):]
        rec['folder']=newrel
        if rec.get('preview','').startswith(oldrel+'/'):
            rec['preview']=newrel+rec['preview'][len(oldrel):]
        changes.append({'from':oldrel,'to':newrel,'files_unchanged':len(before_hash)})
    if changes:dump(output/'folder_length_adjustments.json',changes)
    return changes


def finalize_existing(output):
    summary=load(output/'batch_summary.json')
    if not summary['completed']:raise ValueError('整批尚未完成，不能调整输出目录')
    records=summary['records'];changes=normalize_layout(output,records)
    dump(output/'batch_summary.json',summary)
    write_csv(output/'annotation_index.csv',[r['row'] for r in records if r['status']=='ready_to_annotate'])
    write_csv(output/'review_index.csv',[r['row'] for r in records if r['status']=='needs_review'])
    write_csv(output/'pairs_index.csv',[{k:v for k,v in r.items() if k not in ('row','reasons','compatible_before')} for r in records])
    catalog(output,records)
    shutil.copy2(CODE_ROOT/'build_named_sem_dataset.py',output/'processing_code/build_named_sem_dataset.py')
    print('Folder-length adjustments:',len(changes),flush=True)


def run(source,output):
    start=time.time();cv2.setNumThreads(4);cv2.setRNGSeed(20260913)
    inv=prepare(source,output)
    evidence=screen(inv,output)
    before=[r for r in inv['records'] if r['side']=='before'];after=[r for r in inv['records'] if r['side']=='after']
    edges=evidence['edges'];records=[]
    checkpoint=output/'batch_summary.json'
    if checkpoint.exists():records=load(checkpoint)['records']
    done={r['after'] for r in records};accepted_before=set(r['before'] for r in records if r['status']=='ready_to_annotate')
    for ai,a in enumerate(after):
        if a['path'] in done:continue
        choices=sorted([e for e in edges if e['after_index']==ai],key=lambda e:e['score'],reverse=True)
        best=distinct(choices)
        reverse=distinct([e for e in edges if best and e['before_index']==best['before_index']]) if best else None
        unique=bool(best and reverse and reverse['after_index']==ai)
        rec={'after':a['path'],'status':'unmatched','reason':a['error'] or '没有通过图像几何筛选的同倍率对应图',
             'condition':a['condition'],'source_name':a['original_filename'],
             'compatible_before':[before[i]['path'] for i in evidence['candidates_by_after'].get(str(ai),[])]}
        if choices:
            candidate=best if unique else choices[0];b=before[candidate['before_index']]
            try:
                result=register_arrays(raw_image(b['path']),raw_image(a['path']),
                    b['magnification'],a['magnification'],b['pixel_size_um'],a['pixel_size_um'],
                    matcher=candidate['matcher'],before_content_bounds=b['content_bounds'],after_content_bounds=a['content_bounds'])
                rec=export_pair(output,b,a,result,candidate,unique and b['path'] not in accepted_before)
                if rec['status']=='ready_to_annotate':accepted_before.add(b['path'])
            except Exception as exc:
                rec.update(before=b['path'],status='failed',reason=str(exc))
        records.append(rec)
        dump(checkpoint,{'completed':False,'records':records,'counts':dict(Counter(r['status'] for r in records))})
        print(f'REGISTER {ai+1}/{len(after)} {rec["status"]} {a["condition"]} {a["original_filename"]}',flush=True)
    normalize_layout(output,records)
    ready=[r['row'] for r in records if r['status']=='ready_to_annotate']
    review=[r['row'] for r in records if r['status']=='needs_review']
    write_csv(output/'annotation_index.csv',ready)
    write_csv(output/'review_index.csv',review)
    write_csv(output/'pairs_index.csv',[{k:v for k,v in r.items() if k not in ('row','reasons','compatible_before')} for r in records])
    dump(output/'unpaired_and_failed.json',{'after':[r for r in records if r['status'] in ('unmatched','failed')],
         'before_not_in_ready':[b for b in before if b['path'] not in accepted_before]})
    catalog(output,records)
    summary={'completed':True,'source':str(source),'version':VERSION,'records':records,
             'counts':dict(Counter(r['status'] for r in records)),'seconds':round(time.time()-start,2)}
    dump(checkpoint,summary)
    dump(output/'dataset_manifest.json',{'version':VERSION,'source_inventory':'source_inventory.json',
         'counts':summary['counts'],'methods':'ImageStripSize crop; physical pixel scale; specimen/scale restricted SIFT or LightGlue matching; RANSAC partial affine; ECC with geometric guard',
         'labels':{'before_pore':[0,1],'after_pore':[0,1],'binary':[0,1],
                   'multi':{0:'未变化',1:'新增孔隙',2:'孔隙扩大',3:'孔隙缩小/闭合',4:'新增裂缝',255:'不确定/忽略'}},
         'label_status':'not_started','human_review_status':'pending',
         'limits':'拟合残差不等于独立真值误差。Before/After非重叠仅为候选变化。原始8位仍为8位。',
         'environment':{'python':sys.version,'opencv':cv2.__version__,'numpy':np.__version__}})
    (output/'README.md').write_text(
        '# 带完整底栏 TIFF 生成的 SEM 标注数据集\n\n'
        '双击 `开始标注.bat` 可导入全部自动筛选通过的图对；或在小程序“打开已有任务”选择本数据集目录。\n\n'
        '先查看 `打开数据集总览.html`。文件名包含实验条件、组别、倍率、样品目录和两侧原图序号。\n\n'
        'ready_to_annotate 为自动筛选通过；needs_review 为需复核；都尚未人工验收。双侧孔隙、二分类、多分类标签待人工/SAM标注后保存，空白初始化不是训练真值。\n\n'
        'sources 包含原字节 TIFF/HDR 和底栏。16位主图输出16位PNG，8位源仍输出8位PNG。复核图和SAM输入会做显示归一化。\n\n'
        '读取 ImageStripSize 确定底栏；以物理像素尺寸统一网格，限制同倍率同样品候选、图像验证配对，SIFT/LightGlue + RANSAC + 带几何验收的ECC。详细依据见每对报告与候选清单。\n\n'
        f'自动处理结果：{summary["counts"]}。原始文件、未配对和失败情况可在清单中追溯。\n',encoding='utf-8')
    launcher='@echo off\r\nsetlocal\r\ncall "'+str(CODE_ROOT/'start_integrated_editor.bat')+'" --index "%~dp0annotation_index.csv"\r\n'
    (output/'开始标注.bat').write_bytes(launcher.encode('gbk'))
    code=output/'processing_code';code.mkdir(exist_ok=True)
    for name in ('build_named_sem_dataset.py','registration_pipeline.py','enhanced_matching.py','dataset_paths.py'):
        shutil.copy2(CODE_ROOT/name,code/name)
    print('DONE',json.dumps(summary['counts']),flush=True)
    return summary


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--finalize-only',action='store_true')
    args=parser.parse_args()
    if args.finalize_only:finalize_existing(args.output)
    elif args.prepare_only:prepare(args.source,args.output)
    else:run(args.source,args.output)
