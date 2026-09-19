"""Render the beginner handbook with fixed, audited page breaks and navigation."""
from pathlib import Path
import json,math,re
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph,Frame,Spacer,Table,TableStyle,Preformatted,Image,Flowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor,white
ROOT=Path(__file__).resolve().parents[2];W,H=595.28,841.89
INK=HexColor('#152b3a');TEAL=HexColor('#007e87');MUTED=HexColor('#506673');PALE=HexColor('#eef6f5')
BODY=ParagraphStyle('body',fontName='Helvetica',fontSize=10.3,leading=15.2,textColor=INK,spaceAfter=10)
SMALL=ParagraphStyle('small',parent=BODY,fontSize=8.5,leading=12,spaceAfter=7)
HEAD=ParagraphStyle('subhead',parent=BODY,fontName='Helvetica-Bold',fontSize=12.2,leading=16,textColor=TEAL,spaceBefore=7,spaceAfter=8)
CODE=ParagraphStyle('code',fontName='Courier',fontSize=8.1,leading=11.5,textColor=INK,spaceAfter=10)

def paragraph(t,style=BODY):return Paragraph(t,style)
class Diagram(Flowable):
    def __init__(self,labels,caption):super().__init__();self.labels=labels;self.caption=caption;self.width=W-92;self.height=110
    def draw(self):
        c=self.canv;n=len(self.labels);gap=15;bw=(self.width-gap*(n-1))/n
        for i,label in enumerate(self.labels):
            x=i*(bw+gap);c.setFillColor(PALE);c.setStrokeColor(TEAL);c.roundRect(x,37,bw,63,5,stroke=1,fill=1)
            p=paragraph(label,ParagraphStyle('diagram',parent=SMALL,alignment=1,spaceAfter=0));_,h=p.wrap(bw-12,60);p.drawOn(c,x+6,37+(63-h)/2)
            if i<n-1:
                c.line(x+bw,69,x+bw+gap-2,69);c.line(x+bw+gap-6,72,x+bw+gap-2,69);c.line(x+bw+gap-6,66,x+bw+gap-2,69)
        p=paragraph(self.caption,SMALL);_,h=p.wrap(self.width,30);p.drawOn(c,0,30-h)

def blocks(page):
    out=[]
    for item in page['content']:
        if isinstance(item,str):out.append(paragraph(item))
        elif 'h' in item:out.append(paragraph(item['h'],HEAD))
        elif 'note' in item:
            t=Table([[paragraph(item['note'],SMALL)]],colWidths=[W-92]);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),PALE),('BOX',(0,0),(-1,-1),.5,TEAL),('LEFTPADDING',(0,0),(-1,-1),11),('RIGHTPADDING',(0,0),(-1,-1),11),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),4)]));out.extend([t,Spacer(1,10)])
        elif 'code' in item:out.append(Preformatted(item['code'],CODE,maxLineLength=94))
        elif 'table' in item:
            rows=[[paragraph(escape(str(v)),SMALL) for v in row] for row in item['table']];widths=item.get('widths',[1]*len(rows[0]));widths=[(W-92)*w/sum(widths) for w in widths]
            t=Table(rows,colWidths=widths,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PALE),('LINEBELOW',(0,0),(-1,0),1,TEAL),('LINEBELOW',(0,1),(-1,-1),.3,HexColor('#d8e2e5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),3)]));out.extend([t,Spacer(1,12)])
        elif 'diagram' in item:out.extend([Diagram(item['diagram'],item.get('caption','Conceptual schematic; not anatomical geometry.')),Spacer(1,10)])
        elif 'image' in item:
            image=Image(str(Path(__file__).parent/item['image']));image.drawHeight=image.imageHeight/image.imageWidth*(W-92);image.drawWidth=W-92;out.extend([image,Spacer(1,6),paragraph(item.get('caption',''),SMALL),Spacer(1,6)])
        elif 'small' in item:out.append(paragraph(item['small'],SMALL))
    return out

def render(doc):
    pages=doc['pages'];toc_count=math.ceil(len(pages)/28);toc_size=math.ceil(len(pages)/toc_count);offset=1+toc_count;total=offset+len(pages)
    dest=ROOT/'PLAN'/doc['filename'];dest.parent.mkdir(exist_ok=True)
    c=canvas.Canvas(str(dest),pagesize=(W,H));c.setTitle(doc['title']);c.setAuthor('FruitFlyBrain project');c.setSubject('Beginner handbook, dataset guide, control reference and reproducible setup')
    c.setFillColor(INK);c.rect(0,0,W,H,fill=1,stroke=0);c.setFillColor(HexColor('#65d7ca'));c.setFont('Helvetica-Bold',10);c.drawString(46,H-70,'FRUITFLYBRAIN / THE HANDBOOK')
    title=paragraph('From a fly wiring map<br/>to an experiment<br/>you can understand',ParagraphStyle('cover',fontName='Helvetica-Bold',fontSize=35,leading=42,textColor=white));_,th=title.wrap(W-92,300);title.drawOn(c,46,H-140-th)
    p=paragraph('Neuroscience from first principles. Real dataset examples. Every dashboard control. CPU and GPU setup. Evidence, limitations and a staged research plan.',ParagraphStyle('intro',parent=BODY,fontSize=14,leading=21,textColor=HexColor('#c4dce4')));_,ph=p.wrap(W-115,130);p.drawOn(c,46,345-ph)
    for i,label in enumerate(['UNDERSTAND THE BIOLOGY','TRACE THE DATA','CONTROL THE SIMULATION']):c.setFont('Helvetica',10);c.setFillColor(HexColor('#65d7ca'));c.drawString(46,165-i*25,label)
    c.setFillColor(white);c.setFont('Helvetica',10);c.drawString(46,57,f"Edition {doc['version']} | 19 September 2026 | {total} pages");c.setFont('Helvetica',8);c.drawString(46,38,'An exploratory software model. No demonstrated learning or validated fly behavior.');c.bookmarkPage('cover');c.addOutlineEntry('Cover','cover',0);c.showPage()
    for ti in range(toc_count):
        c.bookmarkPage(f'contents{ti}');c.addOutlineEntry(f'Contents {ti+1}',f'contents{ti}',0)
        decorate(c,'CONTENTS',ti+2,total);c.setFont('Helvetica-Bold',26);c.setFillColor(INK);c.drawString(46,H-96,'A guide to the book' if ti==0 else 'Contents continued')
        c.setFont('Helvetica',10);c.setFillColor(MUTED);c.drawString(46,H-120,'Click a topic or use your PDF reader\'s bookmarks.')
        y=H-157
        for idx,page in list(enumerate(pages))[ti*toc_size:(ti+1)*toc_size]:
            text=f"{page['chapter']} / {page['title']}";c.setFont('Helvetica',9.2);c.setFillColor(INK)
            if c.stringWidth(text,'Helvetica',9.2)>W-130:raise ValueError('TOC title too long: '+text)
            c.drawString(46,y,text);c.drawRightString(W-46,y,str(offset+idx+1));c.linkRect('',f'p{idx}',(42,y-4,W-42,y+12),relative=0,thickness=0);y-=28
        c.showPage()
    previous=None
    for idx,page in enumerate(pages):
        c.bookmarkPage(f'p{idx}')
        if page['chapter']!=previous:c.addOutlineEntry(page['chapter'],f'p{idx}',0);previous=page['chapter']
        c.addOutlineEntry(page['title'],f'p{idx}',1)
        decorate(c,page['chapter'].upper(),offset+idx+1,total)
        title=paragraph(page['title'],ParagraphStyle('title',parent=BODY,fontName='Helvetica-Bold',fontSize=25,leading=30,spaceAfter=0));_,th=title.wrap(W-92,130);title.drawOn(c,46,H-72-th)
        c.setStrokeColor(TEAL);c.setLineWidth(2);c.line(46,H-86-th,W-46,H-86-th)
        flow=blocks(page);top=H-101-th
        fr=Frame(46,66,W-92,top-66,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0);fr.addFromList(flow,c)
        if flow:raise RuntimeError(f"Overflow page {offset+idx+1}: {page['title']} ({len(flow)} blocks)")
        c.showPage()
    c.save();print(f'{dest}: {total} pages');return total

def decorate(c,chapter,page,total):
    c.setFillColor(INK);c.rect(0,H-14,W,14,fill=1,stroke=0)
    c.setFillColor(TEAL);c.setFont('Helvetica-Bold',8.5);c.drawString(46,H-43,'FRUITFLYBRAIN / '+chapter)
    c.setFillColor(MUTED);c.setFont('Helvetica',8);c.drawRightString(W-46,H-43,'HANDBOOK v1.0')
    c.setStrokeColor(HexColor('#d8e2e5'));c.setLineWidth(.5);c.line(46,48,W-46,48)
    c.setFont('Helvetica',8);c.drawString(46,33,'Anatomy, assumptions, engineering and evidence are kept distinct');c.drawRightString(W-46,33,f'{page:02d} / {total:02d}')

if __name__=='__main__':render(json.loads((Path(__file__).parent/'plan.json').read_text()))
