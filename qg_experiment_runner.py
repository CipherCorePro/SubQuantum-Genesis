# qg_experiment_runner.py
import numpy as np
import pandas as pd
import os
import time
import json
from datetime import datetime
from pathlib import Path

# Importiere Simulationskern und Metriken/Helfer
from subqg_core import (
    VortiefenMatrix, SymbolInjector, QuantumNoiseField, MacroWorld,
    predict_future, calculate_fft_dominant_freqs, calculate_shannon_entropy,
    calculate_granger_causality, convert_np # Importiere convert_np
)

# --- Globale Konfiguration für Phase 2 ---
SIM_CONFIG = {
    "matrix_size": 64,
    "base_energy": 0.01,
    "coupling": 0.01,
    "steps": 300, # Anzahl Simulationsschritte
    "cluster_threshold": 0.7,
    "future_steps": 200, # Für Vorhersage
    "region_analysis_size": 10 # Größe der Region für Korrelation/Granger
}

EXPERIMENT_CONFIG = {
    "num_runs_per_group": 5, # Reduziert für Demo, sollte 50-100 sein
    "position_jitter_std": 1.5, # Standardabweichung für Positionsvariation
    "base_seed": 42, # Start-Seed für Reproduzierbarkeit
    "output_dir_base": "run_results"
}

# Definition der Experimentgruppen (Teil 2 - Kontrollgruppen)
EXPERIMENT_GROUPS = {
    "G1_Baseline": {
        "description": "Kein Symbol (Baseline)",
        "symbols": [] # Leere Liste für keine Symbole
    },
    "G2_PlusOnly": {
        "description": "Nur Pluszeichen",
        "symbols": [
            {"name": "Plus", "pos_x": SIM_CONFIG['matrix_size'] // 4, "pos_y": SIM_CONFIG['matrix_size'] // 4, "intensity": 0.03}
        ]
    },
    "G3_CircleOnly": {
        "description": "Nur Kreis",
        "symbols": [
            {"name": "Kreis", "pos_x": 3 * SIM_CONFIG['matrix_size'] // 4, "pos_y": 3 * SIM_CONFIG['matrix_size'] // 4, "intensity": 0.06}
        ]
    },
    "G4_Combined": {
        "description": "Beide Symbole (Plus und Kreis)",
        "symbols": [
            {"name": "Plus", "pos_x": SIM_CONFIG['matrix_size'] // 4, "pos_y": SIM_CONFIG['matrix_size'] // 4, "intensity": 0.03},
            {"name": "Kreis", "pos_x": 3 * SIM_CONFIG['matrix_size'] // 4, "pos_y": 3 * SIM_CONFIG['matrix_size'] // 4, "intensity": 0.06}
        ]
    }
}

# --- Hilfsfunktionen ---

def setup_logging(base_dir):
    """Erstellt das Haupt-Ausgabeverzeichnis mit Zeitstempel."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logging-Verzeichnis erstellt: {run_dir}")
    return run_dir

def run_single_simulation(run_id, group_id, group_config, seed):
    """Führt eine einzelne Simulation durch."""
    print(f"  Starte Run {run_id} für Gruppe {group_id} (Seed: {seed})...")
    np.random.seed(seed) # Wichtig für Reproduzierbarkeit jedes Laufs

    # Initialisiere Simulationsobjekte
    subqg = VortiefenMatrix(size=SIM_CONFIG['matrix_size'], base_energy=SIM_CONFIG['base_energy'])
    quantumfield = QuantumNoiseField(subqg, cluster_threshold=SIM_CONFIG['cluster_threshold'])
    macroworld = MacroWorld()

    # Speichere die tatsächlichen Konfigurationen dieses Laufs
    run_config = {
        "run_id": run_id, "group_id": group_id, "seed": seed,
        **SIM_CONFIG, # Füge globale Sim-Config hinzu
        "symbols_config": [] # Details zu den tatsächlich injizierten Symbolen
    }

    symbol_details = [] # Track details per symbol for this run

    # Injiziere Symbole (mit Jitter)
    for i, symbol_cfg in enumerate(group_config['symbols']):
        symbol_pattern, symbol_size = SymbolInjector.get_symbol(symbol_cfg['name'])
        if symbol_pattern is not None:
            actual_x, actual_y = subqg.inject_symbol(
                symbol_pattern,
                symbol_cfg['pos_x'], symbol_cfg['pos_y'],
                symbol_cfg['intensity'],
                jitter_std=EXPERIMENT_CONFIG['position_jitter_std']
            )
            # Stelle sicher, dass actual_x/y nicht None sind, bevor sie verwendet werden
            if actual_x is not None and actual_y is not None:
                 symbol_info = {
                    **symbol_cfg,
                    "id": i,
                    "actual_pos_x": int(actual_x), # Konvertiere explizit zu int
                    "actual_pos_y": int(actual_y), # Konvertiere explizit zu int
                    "size": int(symbol_size)       # Konvertiere explizit zu int
                 }
                 run_config["symbols_config"].append(symbol_info)
                 symbol_details.append(symbol_info) # Für Metrikberechnung merken
            else:
                # Dieser Fall sollte nicht eintreten, wenn symbol_pattern nicht None ist
                run_config["symbols_config"].append({**symbol_cfg, "id": i, "status": "injection_failed"})
        else:
             run_config["symbols_config"].append({**symbol_cfg, "id": i, "status": "symbol_not_found"})


    # Listen für Zeitreihen
    subqg_region_energies = {f"symbol_{s['id']}_region_avg_E": [] for s in symbol_details}
    macro_signal_history = []

    # Simulationsschleife
    for step in range(SIM_CONFIG['steps']):
        subqg.step(coupling=SIM_CONFIG['coupling'])
        noise = quantumfield.generate_noise()
        macroworld.evolve(noise)
        macro_signal_history.append(macroworld.values[-1])

        # Berechne regionale Energie für Granger etc. (Teil 6)
        for s in symbol_details:
             avg_e = subqg.get_region_average_energy(
                 s.get('actual_pos_x'), # Verwende .get für Sicherheit
                 s.get('actual_pos_y'),
                 SIM_CONFIG['region_analysis_size']
             )
             # Stelle sicher, dass die Spalte existiert, bevor sie gefüllt wird
             col_name = f"symbol_{s['id']}_region_avg_E"
             if col_name in subqg_region_energies:
                 subqg_region_energies[col_name].append(avg_e)


    # Sammle Ergebnisse
    final_energy_matrix = subqg.get_energy_matrix()
    final_noise_map = noise # Letztes Rauschen
    macro_signal = macroworld.get_macro_values()
    macro_prediction = predict_future(macro_signal, steps=SIM_CONFIG['future_steps'])

    # Zeitreihen als DataFrame für Metriken
    # Stelle sicher, dass alle Listen die gleiche Länge haben
    ts_dict = {
        'step': range(SIM_CONFIG['steps']),
        'macro_signal': macro_signal_history
    }
    for col_name, data_list in subqg_region_energies.items():
        if len(data_list) == SIM_CONFIG['steps']:
            ts_dict[col_name] = data_list
        else:
            print(f"Warnung: Länge von {col_name} ({len(data_list)}) stimmt nicht mit Steps ({SIM_CONFIG['steps']}) überein. Überspringe Spalte.")

    timeseries_data = pd.DataFrame(ts_dict)


    return run_config, final_energy_matrix, final_noise_map, macro_signal, macro_prediction, timeseries_data, symbol_details

def calculate_run_metrics(final_energy, macro_signal, timeseries_data, symbol_details):
    """Berechnet Metriken für einen einzelnen Lauf."""
    metrics = {}

    # Metrik C: FFT auf Makrosignal
    metrics['macro_fft_dominant_freqs'], metrics['macro_fft_dominant_amps'] = calculate_fft_dominant_freqs(macro_signal)

    # Metrik D: Entropie-Analyse
    metrics['macro_shannon_entropy'] = calculate_shannon_entropy(macro_signal)

    # Korrelation zwischen Symbolregionen (am Ende)
    region_data = {}
    for s in symbol_details:
        region_id = f"symbol_{s['id']}"
        # Zentrum der Injektion verwenden
        center_x = s.get('actual_pos_x')
        center_y = s.get('actual_pos_y')
        symbol_size = s.get('size', 0) # Größe holen

        if center_x is not None and center_y is not None:
             # Region um das *Zentrum* des Symbols, nicht den Startpunkt
             eff_center_x = (center_x + symbol_size // 2) % SIM_CONFIG['matrix_size']
             eff_center_y = (center_y + symbol_size // 2) % SIM_CONFIG['matrix_size']
             half_size = SIM_CONFIG['region_analysis_size'] // 2

             x_indices = [(eff_center_x + i) % SIM_CONFIG['matrix_size'] for i in range(-half_size, half_size + 1)]
             y_indices = [(eff_center_y + j) % SIM_CONFIG['matrix_size'] for j in range(-half_size, half_size + 1)]

             # Extrahiere Region sicher mit np.ix_
             try:
                region_values = final_energy[np.ix_(x_indices, y_indices)].flatten()
                region_data[region_id] = region_values
             except IndexError as e:
                 print(f"  Warnung Indexfehler bei Regionsextraktion für {region_id}: {e}")
                 region_data[region_id] = np.array([]) # Leeres Array bei Fehler
        else:
             region_data[region_id] = np.array([])

    metrics['symbol_region_correlation'] = {}
    symbols_present = list(region_data.keys())
    if len(symbols_present) >= 2:
        # Berechne Korrelation für alle Paare (hier nur für die ersten beiden)
        s1_key = symbols_present[0]
        s2_key = symbols_present[1]
        if region_data[s1_key].size > 1 and region_data[s2_key].size > 1:
             # Sicherstellen gleicher Länge für corrcoef
             min_len = min(region_data[s1_key].size, region_data[s2_key].size)
             try:
                corr = np.corrcoef(region_data[s1_key][:min_len], region_data[s2_key][:min_len])[0, 1]
                metrics['symbol_region_correlation'][f'{s1_key}_vs_{s2_key}'] = corr if np.isfinite(corr) else None
             except ValueError as e:
                 print(f"  Warnung Korrelation: {e}")
                 metrics['symbol_region_correlation'][f'{s1_key}_vs_{s2_key}'] = None
        else:
             metrics['symbol_region_correlation'][f'{s1_key}_vs_{s2_key}'] = None # Nicht genug Daten

    # Metrik F: Granger-Kausalität
    metrics['granger_causality'] = {}
    for s in symbol_details:
         symbol_region_col = f"symbol_{s['id']}_region_avg_E"
         if symbol_region_col in timeseries_data.columns:
            # Test: Symbolregion -> Makro
            gc_p_value = calculate_granger_causality(
                timeseries_data,
                [symbol_region_col, 'macro_signal'],
                maxlag=5
            )
            metrics['granger_causality'][f'symbol{s["id"]}_E_causes_macro_p_min'] = gc_p_value

            # Test: Makro -> Symbolregion
            gc_p_value_rev = calculate_granger_causality(
                timeseries_data,
                ['macro_signal', symbol_region_col],
                maxlag=5
            )
            metrics['granger_causality'][f'macro_causes_symbol{s["id"]}_E_p_min'] = gc_p_value_rev

    return metrics

def save_run_results(output_path, run_config, final_energy, final_noise, macro_signal, macro_prediction, timeseries_data, metrics):
    """Speichert alle Ergebnisse eines Laufs."""
    output_path.mkdir(parents=True, exist_ok=True)

    # Konfiguration speichern (MIT NumPy Konvertierung)
    try:
        with open(output_path / "run_config.json", 'w') as f:
            json.dump(run_config, f, default=convert_np, indent=4)
    except TypeError as e:
        print(f"!! FEHLER beim Speichern von run_config.json: {e}")
        print(f"   Betroffene Konfiguration: {run_config}") # Zum Debuggen ausgeben


    # CSV Daten speichern (mit float_format für Lesbarkeit)
    try:
        pd.DataFrame(final_energy).to_csv(output_path / "final_subqg_energy.csv", index=False, float_format='%.6e')
        pd.DataFrame(final_noise).to_csv(output_path / "final_noise_map.csv", index=False, float_format='%.6e')
        pd.DataFrame(macro_signal, columns=['macro_value']).to_csv(output_path / "macro_signal.csv", index=False, float_format='%.6e')
        pd.DataFrame(macro_prediction, columns=['prediction']).to_csv(output_path / "macro_prediction.csv", index=False, float_format='%.6e')
        timeseries_data.to_csv(output_path / "timeseries_data.csv", index=False, float_format='%.6e')
    except Exception as e:
        print(f"!! FEHLER beim Speichern von CSV-Dateien: {e}")


    # Metriken speichern (als JSON)
    try:
        with open(output_path / "metrics.json", 'w') as f:
            json.dump(metrics, f, default=convert_np, indent=4)
    except TypeError as e:
        print(f"!! FEHLER beim Speichern von metrics.json: {e}")
        print(f"   Betroffene Metriken: {metrics}") # Zum Debuggen ausgeben

    # print(f"    Ergebnisse gespeichert in: {output_path}") # Auskommentiert für weniger Output

# --- Hauptausführung ---
if __name__ == "__main__":
    start_time_total = time.time()

    # Logging einrichten
    main_output_dir = setup_logging(EXPERIMENT_CONFIG['output_dir_base'])

    meta_log_data = [] # Liste zum Sammeln der Meta-Daten aller Läufe

    run_counter = 0
    # Schleife durch Experimentgruppen
    for group_id, group_config in EXPERIMENT_GROUPS.items():
        print(f"\n--- Starte Gruppe: {group_id} ({group_config['description']}) ---")
        start_time_group = time.time()

        # Schleife für statistische Wiederholungen
        for i in range(EXPERIMENT_CONFIG['num_runs_per_group']):
            run_id = f"{group_id}_run{i+1:03d}"
            current_seed = EXPERIMENT_CONFIG['base_seed'] + run_counter

            try: # Füge try/except um den gesamten Run hinzu
                # 1. Simulation durchführen
                run_config, final_energy, final_noise, macro_signal, macro_prediction, timeseries_data, symbol_details = run_single_simulation(
                    run_id, group_id, group_config, current_seed
                )

                # 2. Metriken berechnen
                run_metrics = calculate_run_metrics(final_energy, macro_signal, timeseries_data, symbol_details)

                # 3. Ergebnisse speichern
                run_output_path = main_output_dir / run_id
                save_run_results(run_output_path, run_config, final_energy, final_noise, macro_signal, macro_prediction, timeseries_data, run_metrics)

                # 4. Meta-Log Eintrag vorbereiten
                meta_entry = {
                    "run_id": run_id,
                    "group_id": group_id,
                    "seed": current_seed,
                    "num_symbols": len(run_config.get('symbols_config', [])),
                    "final_macro_mean": np.mean(macro_signal) if len(macro_signal) > 0 else np.nan,
                    "macro_entropy": run_metrics.get('macro_shannon_entropy', np.nan),
                    **{f"corr_{k}": v for k, v in run_metrics.get('symbol_region_correlation', {}).items()}, # Korrelationen
                    **{f"granger_{k}": v for k, v in run_metrics.get('granger_causality', {}).items()} # Granger p-Werte
                }
                 # Dominante Frequenzen und Amplituden hinzufügen (z.B. die erste)
                dom_freqs = run_metrics.get('macro_fft_dominant_freqs', [])
                dom_amps = run_metrics.get('macro_fft_dominant_amps', [])
                if dom_freqs: meta_entry['macro_dom_freq1'] = dom_freqs[0]
                if dom_amps: meta_entry['macro_dom_amp1'] = dom_amps[0]

                meta_log_data.append(meta_entry)

            except Exception as e:
                 print(f"!!! FEHLER bei Run {run_id}: {e}")
                 # Optional: Hier traceback loggen für detaillierte Fehleranalyse
                 import traceback
                 traceback.print_exc()
                 # Füge einen Fehler-Eintrag zum Meta-Log hinzu (optional)
                 meta_log_data.append({
                    "run_id": run_id,
                    "group_id": group_id,
                    "seed": current_seed,
                    "status": "ERROR",
                    "error_message": str(e)
                 })


            run_counter += 1 # Inkrementiere auch bei Fehler, um Seeds fortzusetzen

        end_time_group = time.time()
        print(f"--- Gruppe {group_id} abgeschlossen ({EXPERIMENT_CONFIG['num_runs_per_group']} Läufe in {end_time_group - start_time_group:.2f} Sek.) ---")


    # Meta-Log als CSV speichern (auch wenn Fehler aufgetreten sind)
    try:
        meta_log_df = pd.DataFrame(meta_log_data)
        meta_log_path = main_output_dir / "meta_log.csv"
        meta_log_df.to_csv(meta_log_path, index=False, float_format='%.6e')
        print(f"\nMeta-Log gespeichert: {meta_log_path}")
    except Exception as e:
        print(f"!!! FEHLER beim Speichern des Meta-Logs: {e}")

    end_time_total = time.time()
    print(f"\n=== Experiment Phase 2 abgeschlossen ===")
    print(f"Gesamtdauer: {end_time_total - start_time_total:.2f} Sekunden")
    print(f"Alle Ergebnisse in: {main_output_dir}")