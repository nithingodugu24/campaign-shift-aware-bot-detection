"""Render report content with reportlab; run with the bundled document runtime."""
from pathlib import Path
import json
from html import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether


def render():
    root = Path.cwd()
    content = json.loads((root / "output/report_content.json").read_text(encoding="utf-8"))
    target = root / "output/Project_Report.pdf"
    width = A4[0] - 100
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", fontName="Helvetica-Bold", fontSize=23, leading=28, spaceAfter=10, textColor=colors.black))
    styles.add(ParagraphStyle(name="ReportHeading", fontName="Helvetica-Bold", fontSize=13, leading=17, spaceBefore=15, spaceAfter=7, keepWithNext=True, textColor=colors.black))
    styles.add(ParagraphStyle(name="ReportBody", fontName="Helvetica", fontSize=9.6, leading=14, spaceAfter=8, allowWidows=0, allowOrphans=0))
    styles.add(ParagraphStyle(name="Cell", fontName="Helvetica", fontSize=7.8, leading=10.5))
    styles.add(ParagraphStyle(name="CellHead", fontName="Helvetica-Bold", fontSize=8, leading=10.5, textColor=colors.white))
    styles.add(ParagraphStyle(name="FigureCaption", fontName="Helvetica", fontSize=8.2, leading=11, spaceBefore=4, spaceAfter=12, textColor=colors.HexColor("#444444")))
    story = [Paragraph(escape(content["title"]), styles["ReportTitle"]), Paragraph(escape(content["subtitle"]), styles["ReportBody"]), Paragraph(content["date"], styles["ReportBody"]), Spacer(1, 8)]
    for block in content["blocks"]:
        kind = block["type"]
        if kind == "heading":
            story.append(Paragraph(escape(block["text"]), styles["ReportHeading"]))
        elif kind == "paragraph":
            story.append(Paragraph(escape(block["text"]), styles["ReportBody"]))
        elif kind == "image":
            img = Image(str(root / block["path"]))
            aspect = img.imageHeight / img.imageWidth
            img.drawWidth = width
            img.drawHeight = width * aspect
            story.append(KeepTogether([img, Paragraph(escape(block["caption"]), styles["FigureCaption"])]))
        else:
            headers, rows = block["headers"], block["rows"]
            data = [[Paragraph(escape(str(x)), styles["CellHead"]) for x in headers]] + [[Paragraph(escape(str(x)), styles["Cell"]) for x in row] for row in rows]
            weights = [max(8, min(26, max([len(str(row[j])) for row in rows] + [len(headers[j])]))) for j in range(len(headers))]
            # Extra room for model/method names; numeric columns remain compact.
            if len(weights) >= 5:
                weights[0] = max(weights[0], 17)
            table = Table(data, colWidths=[width * w / sum(weights) for w in weights], repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#254d5b")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f4f5")]),
                ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#d9d9d9")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
            story.extend([table, Spacer(1, 10)])
    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(50, 28, "Campaign Shift Aware Social Bot Detection | Experimental Report")
        canvas.drawRightString(A4[0] - 50, 28, str(doc.page))
    doc = SimpleDocTemplate(str(target), pagesize=A4, rightMargin=50, leftMargin=50, topMargin=42, bottomMargin=46,
                            title=content["title"], author="Campaign Shift Bot Detection Project", pageCompression=1)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(target)


if __name__ == "__main__":
    render()
