"""Rebuild the research documents from reviewable content; requires reportlab."""
from pathlib import Path
from xml.sax.saxutils import escape
import json, shutil
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Frame, Spacer, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_LEFT

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'PLAN'
INK = HexColor('#142b3b'); TEAL = HexColor('#007e87'); MUTED = HexColor('#526371')
W, H = 595.28, 841.89
BODY = ParagraphStyle('Body', fontName='Helvetica', fontSize=10.2, leading=15.4,
                      textColor=INK, spaceAfter=10)
SUB = ParagraphStyle('Sub', parent=BODY, fontName='Helvetica-Bold', fontSize=12,
                     leading=16, textColor=TEAL, spaceBefore=8, spaceAfter=7)
SMALL = ParagraphStyle('Small', parent=BODY, fontSize=8.5, leading=12, spaceAfter=7)

def p(text, style=BODY):
    return Paragraph(text, style)

def render(doc):
    dest = OUT / doc['filename']
    c = canvas.Canvas(str(dest), pagesize=(W,H), pageCompression=1)
    c.setTitle(doc['title']); c.setAuthor('FruitFlyBrain research project')
    c.setSubject(doc['subtitle'])
    for i, page in enumerate(doc['pages'], 1):
        c.setFillColor(INK); c.rect(0,H-17,W,17,fill=1,stroke=0)
        c.setFillColor(TEAL); c.setFont('Helvetica-Bold',9)
        c.drawString(44,H-43,'FRUITFLYBRAIN / RESEARCH SERIES')
        c.setFont('Helvetica',8); c.setFillColor(MUTED)
        c.drawRightString(W-44,H-43,doc['code']+' | v'+doc.get('version','0.1')+' | '+doc.get('date_label','19 SEP 2026'))
        title = p(page['title'], ParagraphStyle('Title',parent=BODY,fontName='Helvetica-Bold',fontSize=25,leading=29))
        _,th=title.wrap(W-88,120); title.drawOn(c,44,H-70-th)
        c.setStrokeColor(TEAL); c.setLineWidth(2); c.line(44,H-84-th,W-44,H-84-th)
        flow=[]
        for item in page['content']:
            if isinstance(item,str): flow.append(p(item))
            elif 'h' in item: flow.append(p(item['h'],SUB))
            elif 'note' in item:
                t=Table([[p(item['note'],SMALL)]],colWidths=[W-108])
                t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),HexColor('#edf6f6')),('BOX',(0,0),(-1,-1),0.5,TEAL),('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
                flow.extend([t,Spacer(1,12)])
            elif 'refs' in item:
                for n in item['refs']:
                    ref=doc['references'][str(n)]
                    flow.append(p(f'<b>[{n:02d}] {escape(ref[0])}</b><br/><link href="{escape(ref[1])}" color="#007e87">{escape(ref[1])}</link>',SMALL))
        bottom=64; top=H-101-th
        fr=Frame(44,bottom,W-88,top-bottom,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
        fr.addFromList(flow,c)
        if flow: raise RuntimeError(f"Overflow: {doc['code']} page {i} ({len(flow)} blocks)")
        c.setStrokeColor(HexColor('#dce4e9')); c.setLineWidth(.5); c.line(44,47,W-44,47)
        c.setFillColor(MUTED); c.setFont('Helvetica',8)
        c.drawString(44,32,doc.get('footer','Prospective research plan | Evidence and assumptions are separated'))
        c.drawRightString(W-44,32,f'{i:02d} / {len(doc["pages"]):02d}')
        c.showPage()
    c.save()
    print(dest)

if __name__=='__main__':
    import sys
    paths=[Path(__file__).parent/name for name in sys.argv[1:]] if len(sys.argv)>1 else sorted(Path(__file__).parent.glob('*.json'))
    for path in paths:
        doc=json.loads(path.read_text())
        if doc.get('format')=='handbook':
            from build_handbook import render as render_handbook
            render_handbook(doc)
        else:render(doc)
