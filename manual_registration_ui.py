"""Paired landmark selection on cropped, unregistered SEM images."""
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from registration_pipeline import crop_sem_content, read_image_unicode, normalize_registration_image


def choose_landmarks(parent, before_path, after_path, inspect_only=False, crop_footer=True):
    images = [Image.fromarray(normalize_registration_image(
              crop_sem_content(read_image_unicode(p))[0] if crop_footer else read_image_unicode(p)))
              for p in (before_path, after_path)]
    win = tk.Toplevel(parent)
    win.title('放大核对图对（不修改图像）' if inspect_only else '人工辅助配准：选择稳定结构，不选正在变化的孔隙边界')
    win.geometry('1200x760')
    win.transient(parent)
    win.grab_set()
    pairs, pending, result = [], [], []
    kind = tk.StringVar(value='fit')
    status = tk.StringVar()
    ttk.Label(win, text='滚轮缩放，中键拖动；关闭后回批量窗口确认配对。' if inspect_only else '先点Before，再点After对应位置。至少6对拟合点＋3对独立检查点；分散选择。滚轮缩放，中键拖动。').pack(pady=5)
    bar = ttk.Frame(win)
    bar.pack(fill='x')
    ttk.Radiobutton(bar,text='拟合点',variable=kind,value='fit').pack(side='left')
    ttk.Radiobutton(bar,text='独立检查点（不参与拟合）',variable=kind,value='check').pack(side='left')
    ttk.Label(win,textvariable=status).pack()
    body = ttk.Frame(win)
    body.pack(fill='both',expand=True)
    canvases, photos, views = [], [None,None], []
    for side in range(2):
        frame = ttk.Frame(body)
        frame.pack(side='left',fill='both',expand=True)
        ttk.Label(frame,text=('Before','After')[side]).pack()
        canvas = tk.Canvas(frame,bg='#202020',width=570,height=600)
        canvas.pack(fill='both',expand=True)
        canvases.append(canvas)
        views.append({'zoom':1., 'cx':images[side].width/2, 'cy':images[side].height/2, 'drag':None})
    def transform(side):
        c,v,im = canvases[side],views[side],images[side]
        w,h = max(c.winfo_width(),80),max(c.winfo_height(),80)
        scale = min(w/im.width,h/im.height)*v['zoom']
        return scale,w/2-v['cx']*scale,h/2-v['cy']*scale,w,h
    def render(*_):
        for side in range(2):
            scale,ox,oy,w,h = transform(side)
            shown = images[side].transform((w,h),Image.Transform.AFFINE,(1/scale,0,-ox/scale,0,1/scale,-oy/scale),Image.Resampling.BILINEAR)
            photos[side] = ImageTk.PhotoImage(shown)
            c = canvases[side]
            c.delete('all')
            c.create_image(0,0,image=photos[side],anchor='nw')
            entries = [(p[side],i+1,k) for i,(p,k) in enumerate(pairs)]
            if pending and side == 0:
                entries.append((pending[0],len(pairs)+1,pending[1]))
            for (x,y),number,k in entries:
                x,y = ox+x*scale,oy+y*scale
                color = '#00ff88' if k == 'fit' else '#ffaa00'
                c.create_oval(x-4,y-4,x+4,y+4,outline=color,width=2)
                c.create_text(x+9,y-9,text=str(number),fill=color)
        status.set(f"拟合点：{sum(k=='fit' for _,k in pairs)}；检查点：{sum(k=='check' for _,k in pairs)}；下一步点{'After' if pending else 'Before'}")
    def click(event,side):
        if inspect_only:
            return
        if side != (1 if pending else 0):
            return
        scale,ox,oy,_,_ = transform(side)
        point = [(event.x-ox)/scale,(event.y-oy)/scale]
        if not (0 <= point[0] < images[side].width and 0 <= point[1] < images[side].height):
            return
        if side == 0:
            pending.extend((point,kind.get()))
        else:
            pairs.append(([pending[0],point],pending[1]))
            pending.clear()
        render()
    def wheel(event,side):
        scale,ox,oy,_,_ = transform(side)
        x,y = (event.x-ox)/scale,(event.y-oy)/scale
        v = views[side]
        v['zoom'] = min(64.,max(1.,v['zoom']*(1.25 if event.delta>0 else .8)))
        new,_,_,w,h = transform(side)
        v['cx'],v['cy'] = x-(event.x-w/2)/new,y-(event.y-h/2)/new
        render()
    def pan(event,side,start=False):
        v=views[side]
        if not start and v['drag'] is not None:
            scale,*_ = transform(side)
            v['cx'] -= (event.x-v['drag'][0])/scale
            v['cy'] -= (event.y-v['drag'][1])/scale
        v['drag']=(event.x,event.y)
        render()
    def undo():
        if pending:
            pending.clear()
        elif pairs:
            pairs.pop()
        render()
    def finish():
        if inspect_only:
            win.destroy()
            return
        data={key:[p for p,k in pairs if k==key] for key in ('fit','check')}
        if pending or len(data['fit'])<6 or len(data['check'])<3:
            messagebox.showwarning('对应点不足','请完成至少6对拟合点和3对独立检查点。',parent=win)
            return
        result.append(data)
        win.destroy()
    ttk.Button(bar,text='撤销末点/末对',command=undo).pack(side='left',padx=10)
    ttk.Button(bar,text='关闭核对窗口' if inspect_only else '计算并验证配准',command=finish).pack(side='right')
    for side,c in enumerate(canvases):
        c.bind('<Configure>',render)
        c.bind('<Button-1>',lambda e,s=side:click(e,s))
        c.bind('<MouseWheel>',lambda e,s=side:wheel(e,s))
        c.bind('<Button-2>',lambda e,s=side:pan(e,s,True))
        c.bind('<B2-Motion>',lambda e,s=side:pan(e,s))
    parent.wait_window(win)
    return result[0] if result else None
