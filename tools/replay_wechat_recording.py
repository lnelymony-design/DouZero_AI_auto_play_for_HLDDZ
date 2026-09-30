"""Offline replay validator for the WeChat miniapp recognizer.

This diagnostic tool does not click the game. It samples recorded MP4 frames,
reconstructs public actions from hand deltas / remaining counts / visible plays,
and prints any suspicious or illegal candidates for regression debugging.

Example:
    python tools/replay_wechat_recording.py game1.mp4 game2.mp4
"""
import argparse
import cv2
import os
from collections import Counter

from helpers.WechatCardRecognizer import WechatCardRecognizer
r=WechatCardRecognizer()
MAP={str(i):i for i in range(3,10)};MAP.update({'T':10,'J':11,'Q':12,'K':13,'A':14,'2':17,'X':20,'D':30})

def legal(cards):
    if not cards:return False
    m=sorted(MAP[c] for c in cards); n=len(m); d=Counter(m)
    if n==1:return True
    if n==2:return m[0]==m[1] or m==[20,30]
    if n==3:return len(d)==1
    if n==4:return len(d)==1 or (len(d)==2 and 3 in d.values())
    if n==5 and len(d)==2 and sorted(d.values())==[2,3]:return True
    keys=sorted(d)
    def cont(x):return len(x)>1 and all(b-a==1 for a,b in zip(x,x[1:])) and x[-1]<=14
    if len(d)==n and n>=5 and cont(keys):return True
    if all(v==2 for v in d.values()) and len(d)>=3 and cont(keys):return True
    if all(v==3 for v in d.values()) and len(d)>=2 and cont(keys):return True
    cnt=Counter(d.values())
    if n==6 and cnt[4]==1 and (cnt[2]==1 or cnt[1]==2):return True
    if n==8 and ((cnt[4]==1 and cnt[2]==2) or cnt[4]==2):return True
    triples=sorted(k for k,v in d.items() if v==3)
    if len(triples)>=2 and cont(triples):
        singles=sum(1 for v in d.values() if v==1); pairs=sum(1 for v in d.values() if v==2)
        if len(triples)==singles+2*pairs:return True
        if len(triples)==pairs and len(d)==len(triples)*2:return True
    return False

def stable_factory():
 d={}
 def stable(k,v,frames=2):
  if k in d and v==d[k][0]: n=d[k][1]+1
  else:n=1
  d[k]=(v,n);return v if n>=frames else None
 return stable

def hand_rec(img):
 n=r._recognize_rank_band(img,(0.005,0.995,0.60,0.755),40,58,240,0.62)
 j=r._detect_jokers(img,(0.005,0.995,0.60,0.87));return r._merge_cards(n,j)
def compact_jokers(bgr,reg):
 h,w=bgr.shape[:2];x0,x1,y0,y1=int(reg[0]*w),int(reg[1]*w),int(reg[2]*h),int(reg[3]*h);roi=bgr[y0:y1,x0:x1]
 m=r._ink_mask(roi);n,_,stats,_=cv2.connectedComponentsWithStats(m,8);cs=[]
 for i in range(1,n):
  x,y,cw,ch,a=map(int,stats[i])
  if 4<=cw<=35 and 8<=ch<=30 and a>=25:cs.append((x+x0,y+y0,cw,ch,a,x+x0+cw/2,y+y0+ch/2))
 cls=[]
 for comp in sorted(cs,key=lambda z:z[5]):
  for cl in cls:
   if abs(comp[5]-sum(q[5] for q in cl)/len(cl))<=12:cl.append(comp);break
  else:cls.append([comp])
 out=[]
 for cl in cls:
  ys=[q[6] for q in cl];span=max(ys)-min(ys) if len(ys)>1 else 0
  if len(cl)<4 or span<45:continue
  left=min(q[0] for q in cl);right=max(q[0]+q[2] for q in cl);top=min(q[1] for q in cl);bottom=max(q[1]+q[3] for q in cl)
  if right-left>26:continue
  patch=bgr[top:bottom,left:right];hsv=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV);gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
  red=((hsv[:,:,1]>100)&(hsv[:,:,2]>70)&((hsv[:,:,0]<15)|(hsv[:,:,0]>170))).sum();dark=(gray<115).sum();rank='D' if red>max(20,dark*.25) else 'X';out.append((left,rank,1.0))
 return out

def played(img,side,threshold=.64,expected=None):
 if side=='left': reg=(.08,.46,.25,.47) if expected is None or expected<=4 else (.08,.48,.25,.52)
 elif side=='right': reg=(.54,.92,.25,.47) if expected is None or expected<=4 else (.52,.92,.25,.52)
 else: reg=(.32,.68,.38,.62)
 n=r._recognize_rank_band(img,reg,28,50,110,threshold)
 jr=(reg[0],reg[1],reg[2],min(.65,reg[3]+.11));j=compact_jokers(img,jr)
 return r._merge_cards(n,j)
def guided(img,side,count):
 for th in (.68,.64,.60,.56):
  x=played(img,side,th,count)
  if len(x)==count and legal(x):return x
 return ''
def hdiff(before,after):
 b=Counter(before);a=Counter(after)
 if any(a[k]>b[k] for k in a):return None
 m=b-a;out=[]
 for ch in before:
  if m[ch]>0:out.append(ch);m[ch]-=1
 return ''.join(out)

def simulate(vp):
 cap=cv2.VideoCapture(vp);fps=cap.get(cv2.CAP_PROP_FPS);step=max(1,round(fps*.70));idx=0;stable=stable_factory()
 pre=None;pos=None;init=False;bottom=None;confirmed=None;expected=None;sidecycle={'me':'right','right':'left','left':'me'};start={0:'right',1:'me',2:'left'}
 tracked={'left':None,'right':None};recent={'left':('',-999),'right':('',-999)};actions=[];lastpass=set();miss={'left':0,'right':0};hmiss=0;issues=[]
 def add_pass(t,side,why='visual'):
  nonlocal expected
  actions.append((round(t,1),side,'PASS',why));expected=sidecycle[side]
 def sync_to(t,actor,counts,rawh):
  nonlocal expected
  guard=0
  while expected is not None and expected!=actor and guard<3:
   # if skipped opponent count already dropped, cannot call pass
   if expected!='me' and counts.get(expected) is not None and tracked.get(expected) is not None and counts[expected]<tracked[expected]:return False
   if expected=='me' and confirmed and rawh:
    rem=hdiff(confirmed,rawh)
    if rem:return False
   add_pass(t,expected,'inferred');guard+=1
  return expected==actor
 while True:
  ok,img=cap.read()
  if not ok:break
  if idx%step:idx+=1;continue
  t=idx/fps;idx+=1
  rawh=hand_rec(img); ih=stable('ih',rawh,5); lh=stable('lh',rawh,3)
  rc={s:r.recognize_remaining_count(img,s,expected=None) for s in ('left','right')}
  for s in ('left','right'):miss[s]=miss[s]+1 if rc[s] is None else 0
  counts={s:stable('c'+s,rc[s],4) for s in ('left','right')}
  hmiss=hmiss+1 if init and not rawh else 0
  if not init:
   if ih and len(ih)==17:pre=ih
   badge=r.detect_landlord_side(img);mp={'right':0,'me':1,'left':2};cand=stable('badge',mp.get(badge),3)
   if ih and len(ih)==20:cand=1
   elif counts['left']==20:cand=2
   elif counts['right']==20:cand=0
   pc=stable('pos',cand,3)
   if pc is not None:pos=pc
   b=stable('bottom',r.recognize_bottom_cards(img),2)
   if pos==1 and ih and len(ih)==20 and pre:
    x=hdiff(ih,pre)
    if x and len(x)==3:b=x
   if b and len(b)==3:bottom=b
   if pos is not None and ih and len(ih)==(20 if pos==1 else 17) and bottom:
    init=True;confirmed=ih;expected=start[pos];ls=start[pos];tracked={'left':20 if ls=='left' else 17,'right':20 if ls=='right' else 17}
    actions.append((round(t,1),'INIT',pos,ih,bottom,tracked.copy()))
   continue
  # cache play candidates
  for s in ('left','right'):
   p=stable('p'+s,played(img,s),2)
   if p:recent[s]=(p,t)
  self_removed=None;self_final=False
  if lh and confirmed and lh!=confirmed:
   rem=hdiff(confirmed,lh)
   if rem and legal(rem):self_removed=rem
   elif rem:issues.append((round(t,1),'illegal-self',rem))
   elif rem=='':confirmed=lh
  if not self_removed and confirmed and hmiss>=4:
   fc=guided(img,'me',len(confirmed))
   if fc:self_removed=fc;self_final=True
  def oppcand(s):
   old=tracked[s];new=counts[s]
   if old is None:return None
   if new is not None and new<old:drop=old-new;target=new;why='count'
   elif new is None and miss[s]>=4 and old>0:drop=old;target=0;why='badgegone'
   else:return None
   cards,pt=recent[s]
   if not cards or t-pt>3 or len(cards)!=drop or not legal(cards):cards=guided(img,s,drop)
   if cards and len(cards)==drop and legal(cards):return cards,target,why
   return None
  # choose actor expected first then later
  for _ in range(3):
   actor=None;payload=None
   order=[expected,sidecycle.get(expected),sidecycle.get(sidecycle.get(expected))] if expected else []
   for s in order:
    if s=='me' and self_removed:actor='me';payload=self_removed;break
    if s in ('left','right'):
     q=oppcand(s)
     if q:actor=s;payload=q;break
   if not actor:break
   if not sync_to(t,actor,counts,rawh):break
   if actor=='me':
    actions.append((round(t,1),'me',payload,0 if self_final else len(lh)));confirmed='' if self_final else lh;self_removed=None;expected=None if self_final else 'right'
   else:
    cards,new,why=payload;actions.append((round(t,1),actor,cards,new,why));tracked[actor]=new;recent[actor]=('',-999);expected=None if new==0 else sidecycle[actor]
   if expected is None:break
  ps=stable('passes',frozenset(r.detect_pass_sides(img)),2)
  if ps is not None:
   sset=set(ps)
   for _ in range(2):
    if expected in sset and expected not in lastpass:add_pass(t,expected,'visual')
    else:break
   lastpass=sset
 cap.release();return actions,issues

def main():
    parser = argparse.ArgumentParser(description='Replay WeChat Dou Dizhu recordings through the recognition state machine.')
    parser.add_argument('videos', nargs='+', help='MP4 recordings to replay')
    args = parser.parse_args()
    for vp in args.videos:
        print('\n###', os.path.basename(vp))
        acts, issues = simulate(vp)
        for action in acts:
            print(action)
        if issues:
            print('ISSUES')
            for issue in issues:
                print(issue)

if __name__ == '__main__':
    main()
