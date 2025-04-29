# subqg_core.py
import numpy as np
from scipy.stats import entropy as calculate_entropy
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd # Import hinzugefügt für Granger

# --- SubQuantum Genesis Engine ---
class SubQNode:
    def __init__(self, energy=0.0):
        self.energy = energy
        # Phase wird hier nicht aktiv genutzt, aber behalten für Konsistenz
        self.phase = np.random.uniform(0, 2 * np.pi)

    def interact(self, neighbor_energy, coupling):
        delta = coupling * (neighbor_energy - self.energy)
        self.energy += delta
        # Phase update beibehalten, falls später benötigt
        self.phase += delta * 0.1

class VortiefenMatrix:
    def __init__(self, size=32, base_energy=0.01):
        self.size = size
        self.base_energy = base_energy
        # Seed wird extern gesetzt für Reproduzierbarkeit
        self.matrix = [[SubQNode(self.base_energy + np.random.normal(0, 0.001)) for _ in range(size)] for _ in range(size)]

    def step(self, coupling=0.01):
        updates = np.zeros((self.size, self.size))
        current_energy = self.get_energy_matrix() # Effizienter: Matrix einmal holen

        for x in range(self.size):
            for y in range(self.size):
                # Nachbarn mit periodischen Randbedingungen holen
                neighbors_energy = [
                    current_energy[(x - 1) % self.size, y],
                    current_energy[(x + 1) % self.size, y],
                    current_energy[x, (y - 1) % self.size],
                    current_energy[x, (y + 1) % self.size]
                ]
                avg_energy = np.mean(neighbors_energy)
                # Update berechnen basierend auf der Energie *vor* dem Schritt
                updates[x, y] = coupling * (avg_energy - current_energy[x, y])

        # Updates anwenden
        for x in range(self.size):
            for y in range(self.size):
                self.matrix[x][y].energy += updates[x, y]
                # Optional: Phase Update
                # self.matrix[x][y].phase += updates[x, y] * 0.1

    def get_energy_matrix(self):
        return np.array([[node.energy for node in row] for row in self.matrix])

    def inject_symbol(self, symbol_pattern, pos_x, pos_y, intensity=0.02, jitter_std=0):
        if symbol_pattern is None:
             return None, None # Kein Symbol injiziert -> Keine Koordinaten zurückgeben

        # Optional: Position Jitter
        if jitter_std > 0:
             pos_x = int(np.round(pos_x + np.random.normal(0, jitter_std)))
             pos_y = int(np.round(pos_y + np.random.normal(0, jitter_std)))
             # Stelle sicher, dass Positionen innerhalb der Grenzen bleiben (Modulo unten kümmert sich darum)
             # pos_x = np.clip(pos_x, 0, self.size - 1) # Nicht unbedingt nötig wegen Modulo
             # pos_y = np.clip(pos_y, 0, self.size - 1)

        sx, sy = symbol_pattern.shape
        actual_pos_x, actual_pos_y = pos_x, pos_y # Speichere die tatsächliche (ggf. gejitterte) Startposition

        for i in range(sx):
            for j in range(sy):
                xi = (pos_x + i) % self.size
                yj = (pos_y + j) % self.size
                if 0 <= xi < self.size and 0 <= yj < self.size: # Sicherheitscheck (sollte immer wahr sein)
                   self.matrix[xi][yj].energy += symbol_pattern[i, j] * intensity
        return actual_pos_x, actual_pos_y # Gib die tatsächliche Startposition zurück

    def get_region_average_energy(self, center_x, center_y, region_size):
        """Berechnet die durchschnittliche Energie in einer quadratischen Region."""
        # Prüfe, ob Koordinaten gültig sind (Integer erwartet)
        if center_x is None or center_y is None or not isinstance(center_x, (int, np.integer)) or not isinstance(center_y, (int, np.integer)):
             # print(f"Warnung: Ungültige Zentrumskoordinaten ({center_x}, {center_y}) für Regionsenergie.")
             return np.nan # Keine Region, wenn kein (gültiges) Zentrum

        half_size = region_size // 2
        total_energy = 0
        count = 0
        current_energy = self.get_energy_matrix() # Aktuelle Energie holen

        for i in range(-half_size, half_size + 1):
             for j in range(-half_size, half_size + 1):
                 # Berechne Index mit Modulo für periodische Ränder
                 xi = (center_x + i) % self.size
                 yj = (center_y + j) % self.size
                 # Zugriff auf die Energie an der berechneten Position
                 total_energy += current_energy[xi, yj]
                 count += 1

        return total_energy / count if count > 0 else np.nan


# --- Symbol Injector ---
class SymbolInjector:
    @staticmethod
    def plus_symbol(size=5):
        symbol = np.zeros((size, size))
        center = size // 2
        symbol[center, :] = 1
        symbol[:, center] = 1
        return symbol, size

    @staticmethod
    def circle_symbol(size=7):
        symbol = np.zeros((size, size))
        center = size // 2
        radius_sq = (size / 2 - 0.5)**2
        for i in range(size):
            for j in range(size):
                if (i - center)**2 + (j - center)**2 < radius_sq:
                    symbol[i, j] = 1
        return symbol, size

    @staticmethod
    def get_symbol(name, size=None):
        if name == "Plus":
             # Standardgröße wenn keine angegeben
            s = size if size is not None else 5
            return SymbolInjector.plus_symbol(s)
        elif name == "Kreis":
            # Standardgröße wenn keine angegeben
            s = size if size is not None else 7
            return SymbolInjector.circle_symbol(s)
        else:
             return None, 0 # Kein Symbol


# --- Quantum Noise Generator ---
class QuantumNoiseField:
    def __init__(self, base_field, cluster_threshold=0.7):
        self.base_field = base_field # Referenz zur VortiefenMatrix Instanz
        self.threshold = cluster_threshold

    def generate_noise(self):
        field = self.base_field.get_energy_matrix()
        field_min = np.min(field)
        field_max = np.max(field)
        if field_max - field_min < 1e-9:
             normalized_field = np.zeros_like(field)
        else:
            normalized_field = (field - field_min) / (field_max - field_min)

        # Seed wird extern gesetzt
        noise = (np.random.normal(0, 0.001, field.shape) + normalized_field * 0.002)
        # Füge extra Rauschen hinzu, wo die Energie über dem Threshold liegt
        noise[normalized_field > self.threshold] += 0.003
        return noise

# --- Makroskopisches Modell ---
class MacroWorld:
    def __init__(self):
        self.values = []

    def evolve(self, quantum_noise):
        macro_value = np.mean(quantum_noise)
        self.values.append(macro_value)

    def get_macro_values(self):
        return np.array(self.values)

# --- Frequenzbasierte Zukunftsprojektion ---
def predict_future(signal, steps=100, num_dominant=5):
    """Erweiterte FFT-Projektion."""
    n = len(signal)
    if n < 2:
        return np.zeros(steps) # Nicht genug Daten

    fft_coeffs = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(n) # Frequenzen holen

    # Amplituden berechnen (ohne DC)
    # Nur positive Frequenzen betrachten (bis Nyquist)
    positive_freq_indices = range(1, n // 2)
    if not positive_freq_indices: # Falls n < 4
         return np.full(steps, signal[-1] if n > 0 else 0.0) # Letzten Wert oder 0 zurückgeben

    amplitudes = np.abs(fft_coeffs[positive_freq_indices]) / (n/2) # Normieren
    dominant_sorted_indices = np.argsort(amplitudes)[::-1] # Sortiere absteigend nach Amplitude
    # Nehme die Indizes der dominantesten Frequenzen (im Bereich 1 bis n//2)
    dominant_indices_in_fft = [positive_freq_indices[i] for i in dominant_sorted_indices[:num_dominant]]


    prediction = np.zeros(steps)
    time_future = np.arange(n, n + steps) # Zeitpunkte für die Vorhersage

    # DC-Komponente (Mittelwert) hinzufügen
    dc_offset = fft_coeffs[0].real / n
    prediction += dc_offset

    # Dominante Frequenzen addieren
    for idx in dominant_indices_in_fft:
        freq = fft_freqs[idx]
        # Amplitude richtig berechnen (Betrag des Koeffizienten normiert)
        amplitude = np.abs(fft_coeffs[idx]) / (n/2)
        phase = np.angle(fft_coeffs[idx])

        # Beitrag dieser Frequenz zur Vorhersage
        prediction += amplitude * np.cos(2 * np.pi * freq * time_future + phase)

    return prediction

# --- Metrik-Funktionen (könnten auch in metrics_analyzer.py) ---

def calculate_fft_dominant_freqs(signal, num_freqs=5):
    """Gibt die dominantesten Frequenzen und ihre Amplituden zurück."""
    n = len(signal)
    if n < 4: # Benötigt mind. 4 Punkte für sinnvolle FFT ohne DC/Nyquist
        return [], []

    fft_coeffs = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(n)

    # Nur positive Frequenzen (Index 1 bis n//2 - 1) betrachten
    positive_freq_indices = range(1, n // 2)
    amplitudes = np.abs(fft_coeffs[positive_freq_indices]) / (n/2) # Normieren

    # Sortiere die Indizes der positiven Frequenzen nach Amplitude
    dominant_sorted_indices_local = np.argsort(amplitudes)[::-1]
    # Wähle die Top N Indizes aus und konvertiere zurück zu globalen FFT-Indizes
    num_available = len(dominant_sorted_indices_local)
    num_to_take = min(num_freqs, num_available)
    dominant_indices_fft = [positive_freq_indices[i] for i in dominant_sorted_indices_local[:num_to_take]]

    dominant_freqs_values = fft_freqs[dominant_indices_fft]
    # Amplituden an den entsprechenden Indizes holen
    dominant_amps_values = np.abs(fft_coeffs[dominant_indices_fft]) / (n/2)


    return list(dominant_freqs_values), list(dominant_amps_values)

def calculate_shannon_entropy(signal, bins=20):
    """Berechnet die Shannon-Entropie des Signals."""
    if len(signal) < 2: return np.nan
    # Histogramm als Wahrscheinlichkeitsverteilung nutzen
    hist, bin_edges = np.histogram(signal, bins=bins, density=True)
    # Vermeide log(0) durch Hinzufügen einer kleinen Konstante oder Filtern
    non_zero_hist = hist[hist > 0]
    if len(non_zero_hist) == 0: return np.nan
    # Entropie berechnen
    return calculate_entropy(non_zero_hist)

def calculate_granger_causality(data, variables, maxlag=5, verbose=False):
    """Führt Granger-Kausalitätstests durch.
       data: pandas DataFrame mit Zeitreihen als Spalten.
       variables: Liste der zwei zu testenden Variablennamen.
    """
    if len(variables) != 2:
        raise ValueError("Granger Causality benötigt genau zwei Variablen.")
    if variables[0] not in data.columns or variables[1] not in data.columns:
        print(f"Warnung: Variablen {variables} nicht in Daten gefunden. Überspringe Granger.")
        return None

    # Stelle sicher, dass keine NaNs vorhanden sind und genug Daten da sind
    test_data = data[variables].dropna()
    if len(test_data) < maxlag + 10: # Sicherstellen, dass genug Datenpunkte für den Test vorhanden sind
        print(f"Warnung: Nicht genug Datenpunkte ({len(test_data)}) für Granger mit maxlag={maxlag} nach dropna. Überspringe.")
        return None

    try:
        # Führe Test durch: Verursacht Spalte 1 Spalte 0? (Index [1] verursacht [0])
        gc_results = grangercausalitytests(test_data[[variables[1], variables[0]]], maxlag=maxlag, verbose=verbose)

        # Extrahiere p-Werte für den F-Test für jeden Lag
        p_values = [gc_results[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag + 1)]
        # Optional: Gib den minimalen p-Wert oder den p-Wert für den maximalen Lag zurück
        min_p_value = min(p_values) if p_values else np.nan
        return min_p_value # Oder: p_values[-1] für den p-Wert bei maxlag

    except Exception as e:
        # Handle potential LinAlgError oder andere Probleme
        print(f"Fehler bei Granger Causality für {variables} mit maxlag={maxlag}: {e}")
        # print(f"Datenkopf:\n{test_data.head()}") # Zum Debuggen
        # print(f"Datenende:\n{test_data.tail()}") # Zum Debuggen
        return np.nan # Gibt NaN zurück bei Fehler

# --- Hilfsfunktion für JSON Konvertierung (kann global sein) ---
def convert_np(obj):
    """Konvertiert NumPy Typen in Standard Python Typen für JSON."""
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
    if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.complex_, np.complex64, np.complex128)): return {'real': obj.real, 'imag': obj.imag}
    if isinstance(obj, (np.bool_)): return bool(obj)
    if isinstance(obj, (np.void)): return None
    # Falls kein numpy Typ, Fehler werfen, um unerwartete Typen zu erkennen
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')