"""Review image pairs before running batch jobs; workers never touch Tk."""
import copy
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from pathlib import Path
from batch_registration import DEFAULT_OUTPUT, scan_dataset, run_batch, audit_folder


def open_batch_window(parent, audit=False):
    win=tk.Toplevel(parent)
    win.title('检测配准质量（不重新配准）' if audit else '添加文件夹，配准：逐对看图确认')
    win.geometry('1220x850')
    win.transient(parent)
    source=tk.StringVar(value='')
    output=tk.StringVar(value=str(DEFAULT_OUTPUT))
    cropped=tk.BooleanVar(value=True)
    method=tk.StringVar(value='普通 SIFT/ORB')
    status=tk.StringVar(value='选择数据根目录；原图不会改动。')
    details=tk.StringVar()
    choice=tk.StringVar()
    state={'scan':None,'busy':False,'folder':None,'photos':[],'index':None}
    win.batch_is_busy=lambda:state['busy']
    messages=queue.Queue()
    stop=threading.Event()
    controls=[]
    for label,var in [('输入根目录',source),('输出目录',output)]:
        row=ttk.Frame(win,padding=4);row.pack(fill='x')
        ttk.Label(row,text=label,width=12).pack(side='left')
        entry=ttk.Entry(row,textvariable=var);entry.pack(side='left',fill='x',expand=True)
        def pick(v=var):
            folder=filedialog.askdirectory(parent=win,title='选择文件夹',initialdir=v.get() or None)
            if folder:v.set(folder)
        button=ttk.Button(row,text='选择',command=pick);button.pack(side='left')
        controls.extend((entry,button))
    bar=ttk.Frame(win,padding=4);bar.pack(fill='x')
    if not audit:
        check=ttk.Checkbutton(bar,text='输入已经裁掉底栏（不再裁剪）',variable=cropped);check.pack(side='left');controls.append(check)
        combo=ttk.Combobox(bar,textvariable=method,values=['普通 SIFT/ORB','增强 LightGlue（CPU，较慢）'],state='readonly',width=30)
        combo.pack(side='left');controls.append(combo)
    ttk.Label(win,textvariable=status,wraplength=1150).pack(fill='x',padx=8)
    tree=ttk.Treeview(win,columns=('confirmed','before','after','reason'),show='headings',height=9)
    for key,text,width in [('confirmed','状态',100),('before','Before',300),('after','After',330),('reason','说明',380)]:
        tree.heading(key,text=text);tree.column(key,width=width)
    tree.pack(fill='both',expand=True,padx=6)
    scroll=ttk.Scrollbar(win,orient='vertical',command=tree.yview);scroll.pack(side='right',fill='y')
    tree.configure(yscrollcommand=scroll.set)
    if not audit:
        selection_bar=ttk.Frame(win,padding=4);selection_bar.pack(fill='x')
        ttk.Label(selection_bar,text='选中行的Before：').pack(side='left')
        chooser=ttk.Combobox(selection_bar,textvariable=choice,state='readonly',width=95)
        chooser.pack(side='left',fill='x',expand=True);controls.append(chooser)
        ttk.Label(win,textvariable=details,wraplength=1150).pack(fill='x',padx=8)
        preview=ttk.Frame(win);preview.pack(fill='x')
        panels=[]
        for label in ('Before（候选）','After（实验后）'):
            frame=ttk.LabelFrame(preview,text=label);frame.pack(side='left',expand=True,fill='both')
            panel=ttk.Label(frame,anchor='center');panel.pack(fill='both',expand=True)
            panels.append(panel)
    def launch(func,kind):
        state['busy']=True;stop.clear()
        def worker():
            try:messages.put((kind,func()))
            except Exception as exc:messages.put(('error',str(exc)))
        threading.Thread(target=worker,daemon=True).start()
    def progress(text):messages.put(('progress',text))
    def fill():
        tree.delete(*tree.get_children())
        for i,p in enumerate(state['scan']['pairs']):
            tree.insert('', 'end',iid=str(i),values=('已看图确认' if p['confirmed'] else '待确认',
                 Path(p['before']).name if p['before'] else '未指定',Path(p['after']).name,p['reason']))
    def show(_event=None):
        if audit or state['busy'] or not state['scan'] or not tree.selection():return
        i=int(tree.selection()[0]);state['index']=i
        pair=state['scan']['pairs'][i]
        records=state['scan']['inventory']['before']
        names=[p['relative'] for p in records]
        chooser.configure(values=names)
        match=next((p for p in records if p['path']==pair['before']),None)
        choice.set(match['relative'] if match else '')
        state['photos']=[]
        info=[]
        lookup={p['path']:p for group in state['scan']['inventory'].values() for p in group}
        for label,panel,path in zip(('Before','After'),panels,(pair['before'],pair['after'])):
            try:
                with Image.open(path) as im:
                    im=im.convert('RGB');im.thumbnail((530,280))
                photo=ImageTk.PhotoImage(im);state['photos'].append(photo)
                panel.configure(image=photo,text='')
                meta=lookup[path]
                info.append(f"{label}: 倍率={meta.get('magnification') or '未知'}；µm/px={meta.get('pixel_size_um') or '未知'}；{meta.get('scale_source') or '缺标定'} {meta.get('error','')}")
            except Exception as exc:panel.configure(image='',text=f'无预览：{exc}')
        details.set(' | '.join(info))
    def changed(_event=None):
        if state['busy'] or state['index'] is None:return
        p=state['scan']['pairs'][state['index']]
        record=next(r for r in state['scan']['inventory']['before'] if r['relative']==choice.get())
        p.update(before=record['path'],confirmed=False,reason='人工改选；需看图确认')
        i=state['index'];fill();tree.selection_set(str(i));show()
    def confirm():
        if state['busy'] or state['index'] is None:return
        i=state['index'];p=state['scan']['pairs'][i]
        if not p['before']:return
        p.update(confirmed=not p['confirmed'],reason='人工目视确认配对' if not p['confirmed'] else '已撤销确认')
        fill();tree.selection_set(str(i));show()
    if not audit:
        chooser.bind('<<ComboboxSelected>>',changed)
        tree.bind('<<TreeviewSelect>>',show)
        button=ttk.Button(selection_bar,text='确认/撤销此对',command=confirm);button.pack(side='left');controls.append(button)
        def full_view():
            if state['busy'] or state['index'] is None:return
            p=state['scan']['pairs'][state['index']]
            from manual_registration_ui import choose_landmarks
            # Existing inspection dialog provides high zoom/pan. Its cancelled
            # or selected points are not used to alter batch pairing/registration.
            if p['before']:
                choose_landmarks(win,p['before'],p['after'],inspect_only=True,crop_footer=not cropped.get())
        button=ttk.Button(bar,text='放大核对选中图对',command=full_view);button.pack(side='left');controls.append(button)
    bottom=ttk.Frame(win,padding=6);bottom.pack(fill='x')
    def scan():
        if state['busy']:return
        selected=source.get()
        launch(lambda:scan_dataset(selected),'scan')
        status.set('正在扫描文件和标定…')
    def start():
        if state['busy']:return
        out=output.get().strip()
        if not out:
            messagebox.showwarning('输出目录','请选择输出目录',parent=win);return
        if audit:
            src=source.get()
            launch(lambda:audit_folder(src,out,progress,stop.is_set),'result')
        else:
            if not state['scan'] or Path(source.get()).resolve()!=Path(state['scan']['root']):
                messagebox.showwarning('请先扫描','输入目录变化后需要重新扫描。',parent=win);return
            confirmed=sum(p['confirmed'] for p in state['scan']['pairs'])
            if not confirmed:
                messagebox.showwarning('请确认图对','请逐对看图确认，未确认图对不会运行。',parent=win);return
            if not messagebox.askyesno('运行批量配准',f'运行{confirmed}对已确认图像，其余跳过。输出到：\n{out}\n原图不修改，旧Mask不迁移。',parent=win):return
            plan=copy.deepcopy(state['scan']);crop=not cropped.get();matcher='lightglue' if method.get().startswith('增强') else 'classic'
            launch(lambda:run_batch(plan,out,crop,matcher,progress,stop.is_set),'result')
        status.set('正在处理；可以停止后续图对，当前图对会先完成。')
    if not audit:ttk.Button(bottom,text='扫描文件夹／生成候选配对',command=scan).pack(side='left')
    ttk.Button(bottom,text='开始检测' if audit else '配准已确认图对',command=start).pack(side='left',padx=5)
    ttk.Button(bottom,text='停止后续任务',command=stop.set).pack(side='left')
    ttk.Button(bottom,text='打开结果目录',command=lambda:os.startfile(str(state['folder'])) if state['folder'] else None).pack(side='right')
    def poll():
        while not messages.empty():
            kind,value=messages.get_nowait()
            if kind=='progress':status.set(value);continue
            state['busy']=False
            if kind=='error':
                status.set('失败：'+value);messagebox.showerror('任务失败',value,parent=win)
            elif kind=='scan':
                state.update(scan=value,index=None);cropped.set(value['cropped_suggestion']);fill()
                status.set(f"Before {len(value['inventory']['before'])}张，After {len(value['inventory']['after'])}张，排除{len(value['excluded'])}张。文件名仅提供候选，请逐对看图确认。")
            else:
                folder,records=value;state['folder']=folder
                counts={s:sum(r['status']==s for r in records) for s in {r['status'] for r in records}}
                status.set(f'已处理{len(records)}对：{counts}；报告：{folder}')
                if audit:
                    tree.delete(*tree.get_children())
                    for r in records:tree.insert('','end',values=(r['status'],r.get('before'),r.get('after'),r.get('reason','')))
        if win.winfo_exists():win.after(100,poll)
    def close():
        if state['busy']:
            stop.set();messagebox.showinfo('正在停止','当前图对完成后可关闭窗口；已完成结果保留。',parent=win)
        else:win.destroy()
    win.protocol('WM_DELETE_WINDOW',close)
    win.after(100,poll)
    return win
