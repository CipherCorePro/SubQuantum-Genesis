# ⚛️ SubQuantum Genesis

**Simulating the emergence of macro-level structures from coded fluctuations in subquantum fields.**

---

## 🌌 Projektübersicht

Dieses Projekt untersucht die Hypothese, dass Energie auf subquantalem Niveau nicht nur eine skalare Größe ist,
sondern zugleich **strukturierte Information** transportieren kann, die gezielt die Entstehung makroskopischer Ereignisse beeinflusst.

Mittels hochauflösender Simulationen modellieren wir:

-   **SubQuantum Fields** (SubQG): dynamische Netze aus Energiefluktuationen auf kleinster Ebene
-   **Codierte Injektionen** (Symbole): gezielte Strukturvorgaben innerhalb des SubQG
-   **Emergente Makroentwicklung**: Aggregation der Effekte zu makroskopisch beobachtbaren Trends
-   **Statistische und kausale Analysen**: Verifikation des Einflusses durch Messung von Signalverläufen, Entropie, Frequenz und Granger-Kausalität

---

## 🎯 Zielsetzung

-   **Demonstration**, dass gezielt gesetzte subquantale Strukturen die Makrowelt signifikant beeinflussen können
-   **Untersuchung**, ob Energie zugleich **Informationsträger** ist
-   **Analyse**, wie sich codierte Ursprungssignaturen in Rauschen und Makroverhalten fortpflanzen
-   **Entwicklung** neuer Methoden für das Studium von Emergenz und verborgenen Kausalitäten unterhalb der Quantenebene

---

## 🛠️ Technische Umsetzung

-   Programmiert in **Python 3.11+** (getestet mit 3.11/3.12)
-   Nutzung von **NumPy**, **SciPy** und **Statsmodels** für Berechnungen und Statistik
-   Einsatz von **Matplotlib/Seaborn** zur Visualisierung der Analyseergebnisse (im Analyzer-Skript)
-   Verwendung von **ReportLab** zur automatisierten PDF-Berichterstellung
-   Speicherung aller Simulationsdaten, Metriken und Parameter für Reproduzierbarkeit
-   Eigene Skripte für Simulation, Analyse und Reporting

### Hauptmodule

| Modul                          | Funktion                                                               |
| :----------------------------- | :--------------------------------------------------------------------- |
| `subqg_core.py`                | Kernklassen und Funktionen für die Simulation (SubQG, Noise, Makro)    |
| `qg_experiment_runner.py`      | Steuerung von Mehrfachläufen, Konfiguration und Datenspeicherung         |
| `metrics_analyzer.py`          | Statistische Analyse der Ergebnisse (Tests, Plots via Matplotlib)        |
| `qg_report_phase2_generator.py`| Automatische Erzeugung wissenschaftlicher PDF-Berichte (via ReportLab) |

---

## ⚙️ Installation & Nutzung

1.  **Klonen Sie das Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **(Optional, aber empfohlen) Erstellen und aktivieren Sie eine virtuelle Umgebung:**
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```
3.  **Installieren Sie die Abhängigkeiten:**
    ```bash
    pip install numpy pandas scipy statsmodels matplotlib seaborn reportlab
    # Optional: Erstellen Sie eine requirements.txt und verwenden Sie: pip install -r requirements.txt
    ```
4.  **Experimente durchführen (Phase 2):**
    ```bash
    python qg_experiment_runner.py
    ```
    *(Dies führt die in der Konfiguration definierten Läufe durch und speichert die Ergebnisse im `run_results`-Ordner.)*
5.  **Ergebnisse analysieren:**
    ```bash
    # Analysiert den letzten Run automatisch
    python metrics_analyzer.py
    # Oder analysiert einen spezifischen Run
    python metrics_analyzer.py run_results/<timestamp_folder_name>
    ```
    *(Dies erstellt Plots und eine Statistik-Zusammenfassung im Analyseordner.)*
6.  **PDF-Bericht generieren:**
    ```bash
    # Generiert den Bericht für den letzten Run automatisch
    python qg_report_phase2_generator.py
    # Oder generiert den Bericht für einen spezifischen Run
    python qg_report_phase2_generator.py run_results/<timestamp_folder_name>
    ```
    *(Dies erstellt die `qg_report_phase2.pdf`-Datei im analysierten Ergebnisordner.)*

---

## 🔬 Wissenschaftlicher Hintergrund

Das Projekt basiert auf folgenden Schlüsselannahmen:

-   **Emergenz**: Komplexe Makrostrukturen entstehen aus einfachen lokalen Interaktionen (vgl. Anderson 1972, Laughlin 2005).
-   **Subquantal kodierte Information**: Energiefluktuationen könnten Träger spezifischer Anweisungen sein, nicht nur zufällige Variationen.
-   **Makrokausalität durch Substratstruktur**: Die Makrowelt könnte nicht nur von zufälligem Quantenrauschen abhängen, sondern von subquantal strukturierter Vorcodierung.

**Zentrale wissenschaftliche Inspirationen:**

-   P. W. Anderson, *More is Different* (1972)
-   R. B. Laughlin, *A Different Universe* (2005)
-   D. Bohm, *Hidden Variables Theory* (1952)

---

## 📊 Aktueller Stand (Phase 2 Ergebnisse)

-   Systematische Experimente mit Kontrollgruppen belegen **signifikanten Einfluss** von Symbolinjektionen auf den mittleren Makro-Level und die Amplitude dominanter Oszillationen.
-   **Granger-Kausalität** zwischen der Dynamik in den Symbolregionen und dem globalen Makrosignal wurde **stark nachgewiesen**.
-   **Persistente Signaturen** der injizierten Symbole sind sowohl in der finalen Energieverteilung als auch in der **Quanten(Noise)-Map** erkennbar.
-   Ein Framework zur automatisierten Durchführung, Analyse und Berichterstellung wurde erfolgreich implementiert.

---

## 🚀 Nächste Schritte

-   **Komplexere Symbole und dynamische Injektionen** (zeitvariable Strukturen)
-   **Erweiterung auf größere Matrizen und längere Simulationen** zur Prüfung der Skalierbarkeit
-   **Training eines Machine Learning Modells** zur Vorhersage der Makroentwicklung basierend auf SubQG-Mustern
-   **Erforschung von kritischen Schwellen, Synchronisationsmustern und Feedback-Loops** im System

---

## 🧠 Zentrale These (gestützt durch Phase 2)

> Energie auf subquantalem Niveau ist mehr als Energie:
> **Sie ist codierte Information**, die gezielt die Zukunft makroskopischer Systeme **kausal formen kann**.

---

## 📄 Lizenz

Dieses Projekt steht unter der **MIT License** – freie Nutzung und Weiterentwicklung unter Nennung der Quelle.

---

## 📚 Literatur (Auswahl)

-   Anderson, P. W. (1972). *More is Different*. Science, 177(4047), 393–396.
-   Laughlin, R. B. (2005). *A Different Universe: Reinventing Physics from the Bottom Down*. Basic Books.
-   Bohm, D. (1952). *A Suggested Interpretation of the Quantum Theory in Terms of Hidden Variables*. Physical Review, 85(2), 166-193.
-   Granger, C. W. J. (1969). *Investigating Causal Relations by Econometric Models and Cross-spectral Methods*. Econometrica, 37(3), 424-438.
