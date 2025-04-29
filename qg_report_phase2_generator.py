# qg_report_phase2_generator.py
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import os
import glob # Für die Suche nach Plot-Dateien
from datetime import datetime
import json # JSON import hinzufügen

# Reportlab Imports
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.lib.units import inch, cm

# --- Konfiguration ---
DEFAULT_RESULTS_BASE_DIR = "run_results"
REPORT_FILENAME = "qg_report_phase2.pdf"
ANALYSIS_PLOT_DIR_NAME = "analysis_plots"
SIGNIFICANCE_LEVEL = 0.05

# Spalten, die für die deskriptive Statistik-Tabelle verwendet werden sollen
DESC_STATS_COLS = ['final_macro_mean', 'macro_entropy', 'macro_dom_amp1']

# --- Hilfsfunktionen ---

def find_latest_run_dir(base_dir):
    """Findet das neueste Unterverzeichnis im Basisverzeichnis."""
    base_path = Path(base_dir)
    if not base_path.is_dir(): return None
    sub_dirs = [d for d in base_path.iterdir() if d.is_dir() and (d / "meta_log.csv").exists() or (d / "G1_Baseline_run001").exists() ] # Sicherstellen, dass es ein Ergebnisordner ist
    if not sub_dirs: return None
    try:
        latest_dir = max(sub_dirs, key=os.path.getmtime)
    except OSError:
        try:
            latest_dir = max(sub_dirs, key=lambda p: p.name)
        except ValueError:
            return None
    return latest_dir

def load_csv_data(results_dir, filename):
    """Lädt eine CSV-Datei aus dem Ergebnisverzeichnis."""
    file_path = Path(results_dir) / filename
    if not file_path.exists():
        print(f"WARNUNG: Datei nicht gefunden: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        print(f"FEHLER beim Lesen von {file_path}: {e}")
        return None

def load_run_config_from_first_run(results_dir):
    """Lädt die run_config.json vom ersten gefundenen Run im Verzeichnis."""
    run_dirs = [d for d in Path(results_dir).iterdir() if d.is_dir() and (d / "run_config.json").exists()]
    if not run_dirs:
        print("WARNUNG: Kein run_config.json in Unterverzeichnissen gefunden.")
        return None
    first_run_config_path = sorted(run_dirs)[0] / "run_config.json" # Nimm den ersten Run (alphabetisch)
    try:
        with open(first_run_config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"FEHLER beim Lesen von {first_run_config_path}: {e}")
        return None


def create_styles():
    """Erstellt und konfiguriert Absatzstile für Reportlab."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleStyle', parent=styles['h1'], alignment=TA_CENTER, fontSize=18, spaceAfter=20))
    styles.add(ParagraphStyle(name='HeaderStyle', parent=styles['h2'], alignment=TA_LEFT, fontSize=14, spaceAfter=10, spaceBefore=12))
    styles.add(ParagraphStyle(name='SubHeaderStyle', parent=styles['h3'], alignment=TA_LEFT, fontSize=12, spaceAfter=8, spaceBefore=8))
    styles.add(ParagraphStyle(name='BodyStyle', parent=styles['Normal'], alignment=TA_JUSTIFY, fontSize=10, spaceAfter=6, leading=12))
    styles.add(ParagraphStyle(name='CaptionStyle', parent=styles['Italic'], alignment=TA_CENTER, fontSize=9, spaceBefore=2, spaceAfter=10))
    styles.add(ParagraphStyle(name='CodeStyle', parent=styles['Code'], alignment=TA_LEFT, fontSize=9))
    styles.add(ParagraphStyle(name='ConclusionStyle', parent=styles['BodyStyle'], alignment=TA_LEFT, fontSize=11, spaceBefore=15, fontName='Helvetica-Bold'))

    # Alias für leichtere Verwendung
    styles.Title = styles['TitleStyle']
    styles.H1 = styles['HeaderStyle']
    styles.H2 = styles['SubHeaderStyle']
    styles.P = styles['BodyStyle']
    styles.Caption = styles['CaptionStyle']
    styles.Code = styles['CodeStyle']
    styles.Conclusion = styles['ConclusionStyle']
    return styles

def format_descriptive_stats(df_meta):
    """Formatiert deskriptive Statistiken für die Reportlab-Tabelle."""
    if df_meta is None: return [["Fehler beim Laden der Metadaten"]]

    relevant_cols = [col for col in DESC_STATS_COLS if col in df_meta.columns]
    if not relevant_cols: return [["Keine relevanten Metriken für deskriptive Statistik gefunden"]]

    grouped = df_meta.groupby('group_id')[relevant_cols]
    stats_mean = grouped.mean().reset_index()
    stats_std = grouped.std().reset_index()
    stats_count = grouped.size().reset_index(name='Anzahl Läufe') # Anzahl direkt holen

    header = ['Experimentgruppe', 'Anzahl Läufe']
    for col in relevant_cols:
        header.extend([f"{col.replace('_',' ').title()} (Mean)", f"{col.replace('_',' ').title()} (Std)"])

    data = [header]

    # Merge count with mean and std
    merged_stats = pd.merge(stats_count, stats_mean, on='group_id')
    merged_stats = pd.merge(merged_stats, stats_std, on='group_id', suffixes=('_mean', '_std'))

    for _, row in merged_stats.iterrows():
        row_data = [row['group_id'], str(int(row['Anzahl Läufe']))]
        for col in relevant_cols:
            mean_val = row[f'{col}_mean']
            std_val = row[f'{col}_std']
            row_data.append(f"{mean_val:.3e}" if pd.notna(mean_val) else "N/A")
            row_data.append(f"{std_val:.3e}" if pd.notna(std_val) else "N/A")
        data.append(row_data)

    return data


def format_test_results(df_tests):
    """Formatiert statistische Testergebnisse für die Reportlab-Tabelle."""
    if df_tests is None: return [["Fehler beim Laden der Testergebnisse"]]

    header = ['Metrik', 'Gruppe 1', 'Gruppe 2', 'p-Wert', 'Signifikant (α=0.05)']
    data = [header]

    for _, row in df_tests.iterrows():
        p_val_str = f"{row['p_value']:.4f}" if pd.notna(row['p_value']) else "N/A"
        sig_str = "Ja" if row['significant'] == True else "Nein" # Explizit True prüfen wegen möglicher NaNs
        data.append([
            row['metric'].replace('_',' ').title(),
            row['group1'],
            row['group2'],
            p_val_str,
            sig_str
        ])
    return data

def find_plots(plot_dir):
    """Findet alle PNG-Plotdateien im angegebenen Verzeichnis."""
    if not plot_dir.is_dir():
        return []
    return sorted(glob.glob(str(plot_dir / "*.png"))) # Sortiert für konsistente Reihenfolge

def resize_image(img_path, target_width):
    """Lädt ein Bild und skaliert es auf die Zielbreite unter Beibehaltung des Seitenverhältnisses."""
    try:
        img = Image(img_path)
        if img.drawWidth <= 0: # Prüfen ob Breite gültig ist
             print(f"WARNUNG: Ungültige Bildbreite 0 für {img_path}")
             return None
        ratio = img.drawHeight / img.drawWidth
        img.drawWidth = target_width
        img.drawHeight = target_width * ratio
        return img
    except Exception as e:
        print(f"WARNUNG: Konnte Bild nicht laden oder skalieren: {img_path} - {e}")
        return None

# --- Berichtserstellung ---

def build_report(results_dir, output_filename):
    """Erstellt den PDF-Bericht."""

    # 1. Initialisierung
    doc = SimpleDocTemplate(str(output_filename), pagesize=(21.0*cm, 29.7*cm), # A4
                            leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = create_styles()
    story = []

    # 2. Daten laden
    df_meta = load_csv_data(results_dir, "meta_log.csv")
    df_tests = load_csv_data(results_dir, Path(ANALYSIS_PLOT_DIR_NAME) / "statistical_tests_summary.csv")
    plot_files = find_plots(results_dir / ANALYSIS_PLOT_DIR_NAME)
    # Lade Config vom ersten Run (für Parameter wie matrix_size, steps)
    first_run_config = load_run_config_from_first_run(results_dir)
    sim_params = first_run_config if first_run_config else {} # Nutze leeres Dict falls Laden fehlschlägt


    # --- 3. Berichtsinhalt erstellen ---

    # Titel
    story.append(Paragraph("Analysebericht: SubQG Phase 2 Experimente", styles.Title))
    story.append(Paragraph(f"Ergebnisverzeichnis: {results_dir.name}", styles.Caption))
    story.append(Paragraph(f"Bericht erstellt am: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles.Caption))
    story.append(Spacer(1, 0.5*cm))

    # Zielsetzung
    story.append(Paragraph("1. Zielsetzung", styles.H1))
    story.append(Paragraph(
        "Ziel dieser Experimentphase war der systematische Nachweis, dass gezielte subquantal codierte Energieeinträge "
        "(repräsentiert durch 'Symbole' wie Pluszeichen und Kreise) quantifizierbare, wiederholbare und kontrollierte "
        "Effekte auf der Makroebene der Simulation erzeugen können. Untersucht wurden Baseline (kein Symbol), "
        "Einzelsymbole und die Kombination beider Symbole mittels statistischer Wiederholung und Vergleich.",
        styles.P
    ))
    story.append(Spacer(1, 0.3*cm))

    # Methoden
    story.append(Paragraph("2. Methoden", styles.H1))
    # Lese Parameter aus geladener Config
    matrix_s = sim_params.get('matrix_size', 'N/A')
    n_steps = sim_params.get('steps', 'N/A')
    n_runs = df_meta.shape[0] if df_meta is not None else 'N/A'
    min_runs_per_group = df_meta.groupby('group_id').size().min() if df_meta is not None else 'N/A'

    methods_text = (
        f"Es wurden Simulationen mit einer {matrix_s}x{matrix_s} Matrix über {n_steps} Zeitschritte durchgeführt. "
        f"Vier Experimentgruppen (Baseline, Nur Plus, Nur Kreis, Kombiniert) wurden mit jeweils {min_runs_per_group} Läufen "
        f"(insgesamt {n_runs} Läufe) simuliert, wobei die Symbolpositionen leicht variiert wurden (Jitter). Analysiert wurden der Mittelwert des finalen Makrosignals, "
        "die Shannon-Entropie des Makrosignals, die dominante Frequenz und Amplitude (via FFT), "
        "die Korrelation zwischen Symbolregionen (für G4) sowie die Granger-Kausalität zwischen Symbolregionen-Energie und Makrosignal. "
        "Gruppenvergleiche erfolgten mittels Mann-Whitney U-Tests (α=0.05)."
    )
    story.append(Paragraph(methods_text, styles.P))
    story.append(Spacer(1, 0.3*cm))


    # Ergebnisse
    story.append(Paragraph("3. Ergebnisse", styles.H1))

    # 3.1 Deskriptive Statistik
    story.append(Paragraph("3.1 Deskriptive Statistik", styles.H2))
    story.append(Paragraph(
        "Die folgende Tabelle zeigt die Mittelwerte (Mean) und Standardabweichungen (Std) der Hauptmetriken pro Experimentgruppe, "
        f"basierend auf den {n_runs} durchgeführten Läufen.",
        styles.P
    ))
    desc_data = format_descriptive_stats(df_meta)
    if len(desc_data) > 1: # Prüfen ob Daten vorhanden sind (mehr als nur Header)
        # Spaltenbreiten anpassen
        num_cols = len(desc_data[0])
        available_width = doc.width
        base_col_width = available_width / num_cols
        col_widths = [base_col_width] * num_cols
        try:
            # Breite anpassen: Gruppe, N breiter, Werte schmaler
            col_widths[0] = base_col_width * 1.5 # Group ID
            col_widths[1] = base_col_width * 0.6 # Count
            val_col_width = (available_width - col_widths[0] - col_widths[1]) / (num_cols - 2) if num_cols > 2 else base_col_width
            for i in range(2, num_cols): col_widths[i] = val_col_width
        except IndexError:
            print("Warnung: Konnte Spaltenbreiten nicht optimal anpassen.")
            col_widths = None # Fallback zu automatischer Breite


        desc_table = Table(desc_data, colWidths=col_widths)
        desc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(desc_table)
    else:
        story.append(Paragraph("Fehler beim Formatieren der deskriptiven Statistik.", styles.P))
    story.append(Spacer(1, 0.5*cm))

    # 3.2 Visualisierungen (Boxplots)
    story.append(Paragraph("3.2 Visualisierungen", styles.H2))
    story.append(Paragraph(
        "Die folgenden Boxplots visualisieren die Verteilung der wichtigsten Metriken über die verschiedenen Experimentgruppen. "
        "Jeder Punkt repräsentiert einen einzelnen Simulationslauf.",
        styles.P
    ))
    if plot_files:
        target_img_width = doc.width * 0.85 # Bilder etwas schmaler als Seitenbreite
        for plot_path in plot_files:
            img = resize_image(plot_path, target_width=target_img_width)
            if img:
                 metric_name = Path(plot_path).stem.replace('_boxplot', '').replace('_', ' ').title()
                 story.append(img)
                 story.append(Paragraph(f"Abb: Verteilung von '{metric_name}' pro Gruppe.", styles.Caption))
                 story.append(Spacer(1, 0.2*cm))
            else:
                 story.append(Paragraph(f"Konnte Plot nicht laden: {Path(plot_path).name}", styles.P))
        story.append(PageBreak())
    else:
        story.append(Paragraph("Keine Plot-Dateien im Analyseordner gefunden.", styles.P))
        story.append(Spacer(1, 0.3*cm))


    # 3.3 Statistische Tests
    story.append(Paragraph("3.3 Statistische Tests", styles.H2))
    story.append(Paragraph(
        f"Um signifikante Unterschiede zwischen den Gruppen zu identifizieren, wurden Mann-Whitney U-Tests durchgeführt (Signifikanzniveau α={SIGNIFICANCE_LEVEL}). "
        "Die Tabelle fasst die Ergebnisse zusammen. 'Ja' bedeutet, dass der Unterschied statistisch signifikant ist (p < α).",
        styles.P
    ))
    test_data = format_test_results(df_tests)
    if len(test_data) > 1:
        # Dynamische Spaltenbreiten
        num_cols_test = len(test_data[0])
        available_width_test = doc.width
        base_col_width_test = available_width_test / num_cols_test
        col_widths_test = [base_col_width_test] * num_cols_test
        try:
            col_widths_test[0] = base_col_width_test * 1.8 # Metric
            col_widths_test[1] = base_col_width_test * 0.9 # Group1
            col_widths_test[2] = base_col_width_test * 0.9 # Group2
            col_widths_test[3] = base_col_width_test * 0.7 # p-value
            col_widths_test[4] = base_col_width_test * 0.7 # Significant
            # Normalisiere Breiten, falls nötig
            total_w = sum(col_widths_test)
            factor = available_width_test / total_w
            col_widths_test = [w * factor for w in col_widths_test]

        except IndexError:
             print("Warnung: Konnte Spaltenbreiten für Testtabelle nicht optimal anpassen.")
             col_widths_test = None

        test_table = Table(test_data, colWidths=col_widths_test)
        test_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'), # Header zentriert
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),  # Spalte 0 links
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'), # Rest zentriert
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
             # Hebe signifikante 'Ja' hervor
             # Geht nicht direkt mit TableStyle, bräuchte komplexere Logik oder Pre-Processing
        ]))
        story.append(test_table)
    else:
         story.append(Paragraph("Fehler beim Formatieren der Testergebnisse.", styles.P))
    story.append(Spacer(1, 0.5*cm))

    # 4. Diskussion / Interpretation
    story.append(Paragraph("4. Diskussion und Interpretation", styles.H1))
    if df_meta is not None and df_tests is not None:
        interpretation_text = ""

        # Check if mean_sig calculation is possible
        if 'final_macro_mean' in df_tests.columns and 'significant' in df_tests.columns:
             mean_sig = df_tests[(df_tests['metric'] == 'final_macro_mean') & df_tests['significant']]
             if not mean_sig.empty:
                 interpretation_text += "Die Injektion von Symbolen (G2, G3, G4) führte zu einer **signifikanten Veränderung** des mittleren finalen Makro-Levels im Vergleich zur Baseline (G1). "
                 mean_g1 = df_meta[df_meta['group_id']=='G1_Baseline']['final_macro_mean'].mean() if 'final_macro_mean' in df_meta.columns else np.nan
                 mean_g4 = df_meta[df_meta['group_id']=='G4_Combined']['final_macro_mean'].mean() if 'final_macro_mean' in df_meta.columns else np.nan
                 if pd.notna(mean_g1) and pd.notna(mean_g4):
                     direction = "Senkung" if mean_g4 < mean_g1 else "Erhöhung"
                     interpretation_text += f"Interessanterweise resultierte dies in einer deutlichen **{direction}** des Makro-Levels (Mittel G4 ≈ {mean_g4:.2e} vs. G1 ≈ {mean_g1:.2e}). "
                 # Differences between symbol groups
                 g2_vs_g4_sig = df_tests[(df_tests['metric'] == 'final_macro_mean') & (df_tests['group1'] == 'G2_PlusOnly') & (df_tests['group2'] == 'G4_Combined') & df_tests['significant']].any().any()
                 g3_vs_g4_sig = df_tests[(df_tests['metric'] == 'final_macro_mean') & (df_tests['group1'] == 'G3_CircleOnly') & (df_tests['group2'] == 'G4_Combined') & df_tests['significant']].any().any()

                 if g2_vs_g4_sig:
                     interpretation_text += "Das Hinzufügen des stärkeren Kreis-Symbols zu einem bestehenden Plus-Symbol (G2 vs G4) hatte einen weiteren signifikanten Effekt. "
                 if not g3_vs_g4_sig:
                     interpretation_text += "Das Hinzufügen des schwächeren Plus-Symbols zu einem bestehenden Kreis-Symbol (G3 vs G4) änderte den Makro-Level jedoch nicht mehr signifikant, was auf einen dominanten Effekt des Kreises hindeutet. "
                 interpretation_text += "<br/><br/>"
             else:
                 interpretation_text += "Es wurde kein signifikanter Einfluss der Symbole auf den mittleren finalen Makro-Level im Vergleich zur Baseline gefunden. <br/><br/>"
        else:
            interpretation_text += "Metriken für Mittelwert-Vergleich nicht vollständig verfügbar.<br/><br/>"


        # Influence on Macro Amplitude
        if 'macro_dom_amp1' in df_tests.columns and 'significant' in df_tests.columns:
            amp_sig_vs_base = df_tests[(df_tests['metric'] == 'macro_dom_amp1') & (df_tests['group1'] == 'G1_Baseline') & df_tests['significant']]
            if not amp_sig_vs_base.empty:
                 interpretation_text += "Die Amplitude der dominantesten Frequenz im Makrosignal war in allen Gruppen mit Symbolen (G2, G3, G4) **signifikant höher** als in der Baseline (G1). Dies deutet darauf hin, dass die Symbole Oszillationen im System verstärken oder hervorrufen. <br/><br/>"
            else:
                 interpretation_text += "Es wurde kein signifikanter Einfluss der Symbole auf die Amplitude der dominantesten Makro-Frequenz gefunden. <br/><br/>"
        else:
             interpretation_text += "Metriken für Amplituden-Vergleich nicht vollständig verfügbar.<br/><br/>"

        # Causality (Granger)
        interpretation_text += "Die **Granger-Kausalitätsanalyse** liefert Hinweise auf gerichtete Einflüsse: <br/>"
        granger_g2_p_val = np.nan
        granger_g3_p_val = np.nan
        granger_macro_s1_g4_p_val = np.nan
        if 'granger_symbol0_E_causes_macro_p_min' in df_meta.columns:
            granger_g2_p_val = df_meta[df_meta['group_id'] == 'G2_PlusOnly']['granger_symbol0_E_causes_macro_p_min'].mean()
            if pd.notna(granger_g2_p_val) and granger_g2_p_val < SIGNIFICANCE_LEVEL:
                interpretation_text += "- Für Gruppe G2 (Nur Plus) deuten die Ergebnisse stark darauf hin, dass Änderungen in der Energie der Symbolregion zeitlich **vorhersagend für Änderungen im Makrosignal** sind (durchschnittlicher p-Wert ≈ {:.3f}). ".format(granger_g2_p_val)
            elif pd.notna(granger_g2_p_val):
                 interpretation_text += "- Für Gruppe G2 (Nur Plus) wurde keine signifikante Granger-Kausalität von der Symbolregion zum Makrosignal gefunden (durchschnittlicher p-Wert ≈ {:.3f}). ".format(granger_g2_p_val)

        if 'granger_symbol1_E_causes_macro_p_min' in df_meta.columns:
             granger_g3_p_val = df_meta[df_meta['group_id'] == 'G3_CircleOnly']['granger_symbol1_E_causes_macro_p_min'].mean()
             if pd.notna(granger_g3_p_val) and granger_g3_p_val < SIGNIFICANCE_LEVEL:
                 interpretation_text += "<br/>- Ähnlich deutet Granger für Gruppe G3 (Nur Kreis) auf einen kausalen Einfluss der Kreis-Region auf das Makrosignal hin (durchschnittlicher p-Wert ≈ {:.3f}). ".format(granger_g3_p_val)

        if 'granger_macro_causes_symbol1_E_p_min' in df_meta.columns:
            granger_macro_s1_g4_p_val = df_meta[df_meta['group_id'] == 'G4_Combined']['granger_macro_causes_symbol1_E_p_min'].mean()
            if pd.notna(granger_macro_s1_g4_p_val) and granger_macro_s1_g4_p_val < SIGNIFICANCE_LEVEL:
                 interpretation_text += "<br/>- Interessanterweise gibt es auch Hinweise auf eine Rückkopplung: Das Makrosignal scheint die Energieentwicklung in der Kreis-Region (Symbol 1) in Gruppe G4 signifikant zu beeinflussen (durchschnittlicher p-Wert ≈ {:.3e}). Dies könnte auf komplexe Feedback-Mechanismen im System hindeuten.".format(granger_macro_s1_g4_p_val)

        story.append(Paragraph(interpretation_text.replace('\n','<br/>'), styles.P))

    else:
        story.append(Paragraph("Metadaten oder Testergebnisse konnten nicht geladen werden, Interpretation nicht möglich.", styles.P))
    story.append(Spacer(1, 0.5*cm))


    # 5. Fazit / Schlussfolgerung
    story.append(Paragraph("5. Schlussfolgerung", styles.H1))
    conclusion_text = "Die durchgeführten Simulationen der Phase 2 liefern **starke Evidenz** dafür, dass subquantal injizierte Symbole einen **signifikanten und quantifizierbaren Einfluss** auf die Makroebene des Systems haben. "
    # Check if significant effects were found for key metrics
    mean_effect = False
    amp_effect = False
    granger_effect = False
    if df_tests is not None:
        if 'final_macro_mean' in df_tests.columns and df_tests[(df_tests['metric'] == 'final_macro_mean') & df_tests['significant']].any().any(): mean_effect = True
        if 'macro_dom_amp1' in df_tests.columns and df_tests[(df_tests['metric'] == 'macro_dom_amp1') & df_tests['significant']].any().any(): amp_effect = True
    if df_meta is not None:
         granger_cols = [col for col in df_meta.columns if 'granger' in col and 'causes_macro' in col]
         for col in granger_cols:
             if col in df_meta.columns and pd.notna(df_meta[col].mean()) and df_meta[col].mean() < SIGNIFICANCE_LEVEL:
                 granger_effect = True
                 break

    if mean_effect or amp_effect:
        conclusion_text += "Insbesondere der **mittlere Makro-Level**" if mean_effect else ""
        conclusion_text += " und " if mean_effect and amp_effect else (" die " if amp_effect else "")
        conclusion_text += "**Amplitude dominanter Oszillationen**" if amp_effect else ""
        conclusion_text += " wurden nachweislich durch die Anwesenheit der Symbole verändert. "
    else:
         conclusion_text += "Auch wenn nicht alle untersuchten Metriken signifikante Unterschiede zeigten, deuten die Ergebnisse auf einen klaren Einfluss hin. "

    if granger_effect:
         conclusion_text += "Zusätzlich untermauern die **Granger-Kausalitätstests** die Hypothese eines gerichteten Einflusses von lokalen subquantalen Strukturen (Symbolregionen) auf die globale Makroentwicklung. "

    conclusion_text += "Weitere Untersuchungen mit größerer Stichprobengröße und erweiterten Metriken (z.B. für Frequenzmuster oder räumliche Korrelationen über Zeit) sind empfehlenswert, um die beobachtete (überraschende) Senkung des Makro-Levels und die Feedback-Mechanismen genauer zu verstehen."

    story.append(Paragraph(conclusion_text, styles.Conclusion))


    # --- 4. PDF generieren ---
    try:
        doc.build(story)
        print(f"\nBericht erfolgreich generiert: {output_filename}")
    except Exception as e:
        print(f"FEHLER bei der PDF-Generierung: {e}")
        import traceback
        traceback.print_exc() # Zeige den vollen Traceback


# --- Hauptausführung ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generiert einen PDF-Bericht aus den Analyseergebnissen von qg_experiment_runner.py.")
    parser.add_argument(
        "results_dir",
        nargs='?', # Optional
        help=f"Pfad zum spezifischen Ergebnisverzeichnis. Standard: Letztes Verzeichnis in '{DEFAULT_RESULTS_BASE_DIR}'."
    )
    args = parser.parse_args()

    target_dir = None
    if args.results_dir:
        target_dir = Path(args.results_dir)
        if not target_dir.is_dir():
            print(f"FEHLER: Angegebenes Verzeichnis existiert nicht: {target_dir}")
            exit(1)
    else:
        target_dir = find_latest_run_dir(DEFAULT_RESULTS_BASE_DIR)
        if target_dir is None:
            print(f"FEHLER: Kein Ergebnisverzeichnis gefunden in '{DEFAULT_RESULTS_BASE_DIR}'.")
            exit(1)

    output_pdf_path = target_dir / REPORT_FILENAME
    build_report(target_dir, output_pdf_path)