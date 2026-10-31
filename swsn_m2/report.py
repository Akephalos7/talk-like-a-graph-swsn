"""Build a two-page reproducible Milestone 2 report using NetworkX results."""
from __future__ import annotations
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm

NAVY = colors.HexColor("#11263E")
BLUE = colors.HexColor("#2563EB")
GREY = colors.HexColor("#546277")

def build_report(df, path: Path, figures: Path, demo=False):
    path = Path(path)
    figures = Path(figures)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=17*mm, bottomMargin=16*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleM2", parent=styles["Title"], fontName="Helvetica-Bold",
                              fontSize=16, leading=20, textColor=NAVY, spaceAfter=10))
    styles.add(ParagraphStyle(name="HeadingM2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                              fontSize=10.5, leading=14, textColor=NAVY, spaceBefore=10, spaceAfter=7))
    styles.add(ParagraphStyle(name="SmallM2", parent=styles["BodyText"], fontName="Helvetica",
                              fontSize=8, leading=11, textColor=GREY, spaceAfter=8))
    story = []
    P = lambda text, style="SmallM2": Paragraph(text, styles[style])
    def heading(title): story.append(P(title, "HeadingM2"))
    def table(rows, widths):
        t = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),NAVY), ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#EFF3F8"), colors.white]),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,-1),7.3),
            ("BOTTOMPADDING",(0,0),(-1,-1),7), ("TOPPADDING",(0,0),(-1,-1),7),
            ("LINEBELOW",(0,-1),(-1,-1),0.5,colors.HexColor("#D6DFE9")),
        ]))
        story.append(t)
    story.append(P("Milestone 2 | Dataset Selection &amp; EDA", "TitleM2"))
    story.append(P("Ozan Polat &amp; Baris Surmelioglu | 9 November 2026"))
    if demo: story.append(P("<b>DEMONSTRATION ONLY — not real SNAP analysis.</b>"))
    heading("1. Research objective")
    story.append(P("We study how graph structure may affect textual graph encodings and later small-LLM reasoning. Two real social networks and three synthetic graph families are profiled to prepare controlled experiments."))
    heading("2. Dataset selection")
    table([
       ["Dataset","Nodes","Edges","Type"],
       ["Facebook (SNAP)","4,039","88,234","Undirected"],
       ["Email-Eu-core (SNAP)","1,005","25,571","Directed"],
       ["ER / BA / SBM","300 each","Variable","Undirected"]
    ],[175,75,80,135])
    story.append(Spacer(1,7))
    story.append(P("Original SNAP data are downloaded by the analysis code and are not redistributed in this repository."))
    heading("3. Exploratory network measurements")
    rows=[["Network","Nodes","Edges","Mean degree","Density","Components"]]
    for _,r in df.iterrows():
        rows.append([str(r["dataset"]).replace(" synthetic (n=300)","").replace(" (SNAP)",""),
            str(int(r["nodes"])),f'{int(r["edges"]):,}',f'{float(r["mean_total_degree"]):.1f}',
            f'{float(r["density"]):.3f}',str(int(r["weak_or_connected_components"]))])
    table(rows,[135,55,70,75,65,65])
    story.append(Spacer(1,7))
    story.append(P("Components are weakly connected for the directed email graph. Total degree for directed networks includes incoming and outgoing ties."))
    heading("4. Initial observations")
    if demo:
        story.append(P("The demonstration uses synthetic networks and Karate Club only; actual SNAP results are not claimed."))
    else:
        fb=df.loc[df["dataset"]=="Facebook (SNAP)"].iloc[0]
        em=df.loc[df["dataset"]=="Email-Eu-core (SNAP)"].iloc[0]
        story.append(P(f"Facebook: {int(fb['nodes']):,} nodes, {int(fb['edges']):,} edges, mean degree {float(fb['mean_total_degree']):.1f}."))
        story.append(P(f"Email-Eu-core: {int(em['weak_or_connected_components'])} weakly connected components, {int(em['strong_components'])} strongly connected components and {int(em['self_loops'])} self-loops. Self-loops were kept in EDA counts."))
        story.append(P(f"Email has {int(em['department_count'])} anonymized departments; {100*float(em['within_department_edge_fraction']):.1f}% of directed edges are within departments (including self-loops)."))
    story.append(PageBreak())
    story.append(P("Visual Analysis &amp; Next Steps", "TitleM2"))
    heading("5. Degree distributions")
    f=figures/"degree_distributions.png"
    if f.exists(): story.append(Image(str(f),width=157*mm,height=63*mm,kind="proportional"))
    story.append(P("Figure 1. Empirical complementary cumulative degree distributions. The larger real networks show broader degree tails than the 300-node synthetic controls. Network sizes differ, so this comparison is descriptive."))
    heading("6. Representative induced subgraphs")
    f=figures/"sample_networks.png"
    if f.exists(): story.append(Image(str(f),width=157*mm,height=64*mm,kind="proportional"))
    story.append(P("Figure 2. Small local induced subgraphs sampled from the original networks. Layout coordinates are illustrative and are not graph features."))
    heading("7. Experimental preparation")
    story.append(P("Maintain edge direction and 42 department labels for Email-Eu-core. The 642 self-loops are preserved in EDA statistics but excluded from later neighbor questions."))
    story.append(P("The sampling pipeline prepares 60 fixed-seed subgraphs (12, 25 and 50 nodes) and 60 NetworkX reference questions for future comparisons of edge lists, adjacency lists and natural-language descriptions. Sampling may over-represent dense neighborhoods."))
    heading("References")
    story.append(P("[1] McAuley &amp; Leskovec (NeurIPS 2012), SNAP ego-Facebook. [2] Yin et al. (KDD 2017), Email-Eu-core. [3] Fatemi et al. (ICLR 2024), Talk Like a Graph."))
    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica",7)
        canvas.setFillColor(GREY)
        canvas.drawString(18*mm,10*mm,"Semantic Web and Social Networks | M2")
        canvas.drawRightString(192*mm,10*mm,f"Page {document.page}")
        canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return path
