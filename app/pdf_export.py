"""Renders a single TestResult as a downloadable PDF report."""
import io
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                                 TableStyle)

from .disease_content import DISEASE_INFO

BRAND = colors.HexColor("#1c9fb3")
INK = colors.HexColor("#14212b")
MUTED = colors.HexColor("#5c7480")
RISK_HIGH = colors.HexColor("#dc2626")
RISK_LOW = colors.HexColor("#16a34a")

NEGATIVE_LABELS = {"benign", "negative", "no_stone"}


def build_result_pdf(result, disease_key: str) -> bytes:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Title"], textColor=INK, fontSize=18)
    h2_style = ParagraphStyle("h2", parent=styles["Heading2"], textColor=INK, spaceBefore=14)
    body_style = ParagraphStyle("body", parent=styles["BodyText"], textColor=INK)
    muted_style = ParagraphStyle("muted", parent=styles["BodyText"], textColor=MUTED, fontSize=9)

    is_risk = result.prediction.lower() not in NEGATIVE_LABELS
    result_color = RISK_HIGH if is_risk else RISK_LOW

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=22 * mm, bottomMargin=18 * mm, leftMargin=20 * mm, rightMargin=20 * mm,
    )

    story = [
        Paragraph("Multi-Disease Detection System", title_style),
        Paragraph("Preliminary screening report", muted_style),
        Spacer(1, 14),
        Paragraph(
            "Educational preliminary-screening tool only — not a certified diagnostic device. "
            "This result should be discussed with a qualified healthcare professional.",
            ParagraphStyle("disclaimer", parent=body_style, textColor=BRAND, fontSize=9,
                           borderColor=BRAND, borderWidth=0.5, borderPadding=8,
                           backColor=colors.HexColor("#e3f7f9")),
        ),
        Spacer(1, 16),
        Paragraph(
            DISEASE_INFO.get(disease_key, {}).get(
                "label", "Kidney Stone (CT Scan)" if disease_key == "kidney_stone" else disease_key,
            ),
            h2_style,
        ),
    ]

    summary_rows = [
        ["Result", result.prediction.capitalize()],
        ["Confidence", f"{result.probability * 100:.1f}%"],
        ["Model used", result.model_used],
        ["Date", result.created_at.strftime("%Y-%m-%d %H:%M UTC")],
    ]
    summary_table = Table(summary_rows, colWidths=[45 * mm, 100 * mm])
    summary_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
        ("TEXTCOLOR", (1, 0), (1, 0), result_color),
        ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#e1eef0")),
    ]))
    story.append(summary_table)

    story.append(Paragraph("Input panel", h2_style))
    input_rows = [["Field", "Value"]] + [
        [str(k), "" if v is None else str(v)] for k, v in (result.input_data or {}).items()
    ]
    input_table = Table(input_rows, colWidths=[75 * mm, 70 * mm])
    input_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3fbfc")),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e1eef0")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(input_table)

    story.append(Spacer(1, 20))
    story.append(Paragraph(
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} — "
        "educational demonstration only, not a certified medical record.",
        muted_style,
    ))

    doc.build(story)
    return buf.getvalue()
