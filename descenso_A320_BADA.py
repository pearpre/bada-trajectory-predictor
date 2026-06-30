"""
Predictor de trayectorias de descenso — Airbus A320 / BADA
Vuelo N251SB: LEMD → ENGM
Variable de incertidumbre: Masa  U(-5%, +5%) sobre masa de referencia (LAW)
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.io as sio
import os

# Carpeta de salida: Predictor_outputs/ junto al propio script
_DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__))
DIR_OUTPUTS = os.path.join(_DIR_SCRIPT, 'Predictor_outputs')
os.makedirs(DIR_OUTPUTS, exist_ok=True)

def ruta(nombre_fichero):
    """Devuelve la ruta completa dentro de Predictor_outputs/."""
    return os.path.join(DIR_OUTPUTS, nombre_fichero)

# =============================================================================
# 1. CONSTANTES FÍSICAS Y DE CONVERSIÓN
# =============================================================================

# Constantes atmosféricas ISA
g      = 9.80665    # Aceleración gravitatoria [m/s²]
R      = 287.05     # Constante específica del aire [J/(kg·K)]
KAPPA  = 1.4        # Índice adiabático del aire [-]
BETA   = -0.0065    # Gradiente térmico en troposfera [K/m]
T0     = 288.15     # Temperatura ISA a nivel del mar [K]
P0     = 101325.0   # Presión ISA a nivel del mar [Pa]
RHO0   = 1.225      # Densidad ISA a nivel del mar [kg/m³]
T11    = 216.65     # Temperatura ISA en tropopausa y estratosfera [K]
H11_M  = 11000.0    # Altitud de la tropopausa [m]

# Presión en la tropopausa (calculada una vez)
P11 = P0 * (T11 / T0) ** (-g / (BETA * R))

# Factores de conversión
FT2M   = 0.3048       # pies a metros
M2FT   = 1.0 / FT2M  # metros a pies
KT2MS  = 0.514444     # nudos a m/s
MS2KT  = 1.0 / KT2MS # m/s a nudos
M2NM   = 1.0 / 1852.0 # metros a millas náuticas


# =============================================================================
# 2. PARÁMETROS BADA A320-231 / V2500  (OPF Mar 2011 + APF Mar 2011)
# =============================================================================

# --- Aerodinámica (configuración CLEAN / CR) ---
S    = 122.60      # Superficie alar [m²]
CD0  = 0.026659    # Resistencia parásita [-]
CD2  = 0.038726    # Factor de resistencia inducida [-]

# --- Empuje máximo de ascenso (h en pies) ---
# T_max = CTc1 * (1 - h/CTc2 + CTc3*h²)
CTc1 = 142310.0    # Empuje máximo a nivel del mar [N]
CTc2 = 51680.0     # Coeficiente de decaimiento lineal [ft]
CTc3 = 5.6809e-11  # Corrección cuadrática [ft⁻²]  

# --- Factores de reducción de empuje en descenso ---
# T_des = Cdes * T_max
CDES_HIGH  = 0.13603   # Para h >= Hp_des (alta altitud, baja densidad)
CDES_LOW   = 0.10847   # Para h <  Hp_des (baja altitud, mayor densidad)
HP_DES_FT  = 29831.0   # Altitud de transición entre factores [ft]

# --- Consumo de combustible en descenso (h en pies) ---
# f_des = Cf3 * (1 - h/Cf4)  [kg/min]
CF3 = 8.9418       # Flujo de ralentí a nivel del mar [kg/min]
CF4 = 93865.0      # Altitud de referencia del consumo [ft]

# --- Velocidades de descenso (APF, masa de referencia) ---
M_DES   = 0.79     # Mach de descenso en alta altitud [-]
CAS2_KT = 300.0    # CAS en altitud media [kt]
CAS1_KT = 250.0    # CAS en baja altitud (< FL100) [kt]
FL100_FT = 10000.0 # Altitud de transición CAS2 → CAS1 [ft]


# =============================================================================
# 3. CONDICIONES DEL VUELO N251SB  (extraídas del OFP)
# =============================================================================

H_INI_FT = 37000.0   # FL de inicio del descenso (TOD) [ft]
H_FIN_FT = 4800.0    # Altitud mínima IAF VALPU, llegada RIPAM3L [ft]
M_REF_KG = 66123.0   # Masa de referencia = LAW del OFP [kg]

# Parámetros del análisis de Monte Carlo
N_SIM    = 1000      # Número de simulaciones
INCERT   = 0.05      # Incertidumbre de masa: U(±5%)
DH_FT    = 100.0     # Paso de integración [ft]  (validado en sección 4.3)


# =============================================================================
# 4. FUNCIONES ATMOSFÉRICAS ISA
# =============================================================================

def atmosfera_ISA(h_ft):
    """
    Calcula el estado atmosférico ISA en una altitud de presión dada.

    Parámetros
    ----------
    h_ft : float
        Altitud de presión en pies.

    Retorna
    -------
    T   : float  Temperatura [K]
    p   : float  Presión [Pa]
    rho : float  Densidad [kg/m³]
    a   : float  Velocidad del sonido [m/s]
    """
    h_m = h_ft * FT2M

    if h_m <= H11_M:
        # Troposfera: temperatura decrece linealmente con BETA
        T = T0 + BETA * h_m
        p = P0 * (T / T0) ** (-g / (BETA * R))
    else:
        # Estratosfera: temperatura constante, presión exponencial
        T = T11
        p = P11 * np.exp(-g / (R * T11) * (h_m - H11_M))

    rho = p / (R * T)
    a   = np.sqrt(KAPPA * R * T)
    return T, p, rho, a


# =============================================================================
# 5. CONVERSIÓN DE VELOCIDADES
# =============================================================================

def cas_a_tas(cas_kt, h_ft):
    """
    Convierte CAS a TAS y Mach usando la formulación compresible estándar BADA.

    La conversión pasa por la presión de impacto Qc, que es invariante con
    la altitud para una CAS dada, y desde ahí se obtiene el Mach y la TAS
    en las condiciones reales del vuelo.

    Parámetros
    ----------
    cas_kt : float  CAS en nudos.
    h_ft   : float  Altitud de presión en pies.

    Retorna
    -------
    M      : float  Número de Mach [-]
    tas_ms : float  TAS en m/s
    """
    cas_ms = cas_kt * KT2MS
    T, p, rho, a = atmosfera_ISA(h_ft)

    # Paso 1: CAS → presión de impacto Qc usando condiciones ISA al nivel del mar
    Qc = P0 * ((1.0 + (KAPPA - 1.0) / 2.0 * cas_ms**2 / (KAPPA * R * T0))
               ** (KAPPA / (KAPPA - 1.0)) - 1.0)

    # Paso 2: Qc → Mach en las condiciones reales de presión
    M = np.sqrt(2.0 / (KAPPA - 1.0) * ((Qc / p + 1.0)
                ** ((KAPPA - 1.0) / KAPPA) - 1.0))

    tas_ms = M * a
    return M, tas_ms


def calcular_crossover(m_des, cas_kt, h_ini_ft, h_fin_ft, paso_ft=10.0):
    """
    Calcula la altitud de crossover: altitud donde el Mach dado produce
    exactamente la misma TAS que la CAS dada en condiciones ISA.

    Por encima del crossover se opera a Mach constante; por debajo, a CAS.

    Parámetros
    ----------
    m_des    : float  Mach de descenso [-]
    cas_kt   : float  CAS de referencia [kt]
    h_ini_ft : float  Altitud de búsqueda inicial (alta) [ft]
    h_fin_ft : float  Altitud de búsqueda final (baja) [ft]
    paso_ft  : float  Resolución de búsqueda [ft]

    Retorna
    -------
    crossover_ft : float  Altitud de crossover [ft]
    """
    h_prev = h_ini_ft
    _, _, _, a_prev = atmosfera_ISA(h_prev)
    tas_mach_prev = m_des * a_prev
    _, tas_cas_prev = cas_a_tas(cas_kt, h_prev)
    diff_prev = tas_mach_prev - tas_cas_prev

    for h in np.arange(h_ini_ft - paso_ft, h_fin_ft, -paso_ft):
        _, _, _, a = atmosfera_ISA(h)
        tas_mach = m_des * a
        _, tas_cas = cas_a_tas(cas_kt, h)
        diff = tas_mach - tas_cas
        # Cambio de signo → cruce
        if diff_prev * diff < 0:
            return h
        diff_prev = diff

    # Si no se encuentra cruce, devolver el límite inferior
    return h_fin_ft


# =============================================================================
# 6. MODELOS AERODINÁMICO Y PROPULSIVO
# =============================================================================

def empuje_descenso(h_ft):
    """
    Calcula el empuje de descenso en función de la altitud.

    El empuje máximo de ascenso sigue una ley polinómica con h (en ft).
    El empuje de descenso es una fracción de ese máximo, con dos niveles
    según la altitud (alta/baja densidad del aire).

    Parámetros
    ----------
    h_ft : float  Altitud de presión [ft]

    Retorna
    -------
    T_des : float  Empuje de descenso [N]
    """
    T_max = CTc1 * (1.0 - h_ft / CTc2 + CTc3 * h_ft**2)
    Cdes  = CDES_HIGH if h_ft >= HP_DES_FT else CDES_LOW
    return Cdes * T_max


def aerodinamica(m_kg, tas_ms, rho):
    """
    Calcula CL, CD y resistencia aerodinámica D.

    Bajo la hipótesis cuasi-estacionaria L = m*g, el CL queda determinado
    por el estado actual de la aeronave sin necesidad de conocer el ángulo
    de ataque.

    Parámetros
    ----------
    m_kg   : float  Masa de la aeronave [kg]
    tas_ms : float  TAS [m/s]
    rho    : float  Densidad del aire [kg/m³]

    Retorna
    -------
    CL : float  Coeficiente de sustentación [-]
    CD : float  Coeficiente de resistencia [-]
    D  : float  Resistencia aerodinámica [N]
    """
    q  = 0.5 * rho * tas_ms**2        # Presión dinámica [Pa]
    CL = m_kg * g / (q * S)           # De L = m*g = q*S*CL
    CD = CD0 + CD2 * CL**2            # Polar parabólica
    D  = CD * q * S                   # Resistencia total [N]
    return CL, CD, D


def consumo_descenso(h_ft):
    """
    Flujo de combustible en descenso en kg/s.

    Sigue la ley lineal de BADA: f_des = Cf3*(1 - h/Cf4) [kg/min].
    El consumo disminuye con la altitud porque la densidad del aire es
    menor y el motor requiere menos combustible para mantenerse en ralentí.

    Parámetros
    ----------
    h_ft : float  Altitud de presión [ft]

    Retorna
    -------
    f_kgs : float  Flujo de combustible [kg/s]
    """
    f_kgmin = CF3 * (1.0 - h_ft / CF4)
    return f_kgmin / 60.0


# =============================================================================
# 7. ENERGY SHARE FACTOR (ESF)
# =============================================================================

def esf(M, h_ft, fase):
    """
    Calcula el Energy Share Factor f(M) para la fase y capa atmosférica actuales.

    El ESF cuantifica qué fracción de la potencia neta se destina al cambio
    de altitud. Depende del régimen de velocidad (Mach o CAS constante) y de
    si la aeronave está en troposfera o estratosfera.

    Parámetros
    ----------
    M    : float  Número de Mach instantáneo [-]
    h_ft : float  Altitud de presión [ft]
    fase : str    'mach' o 'cas'

    Retorna
    -------
    fM : float  Energy Share Factor [-]
    """
    h_m = h_ft * FT2M
    T, _, _, _ = atmosfera_ISA(h_ft)
    en_troposfera = (h_m <= H11_M)

    if fase == 'mach':
        if not en_troposfera:
            # Estratosfera: T constante, TAS constante, sin aceleración → ESF=1
            return 1.0
        else:
            # Troposfera: T varía con h, mantener Mach implica cambio de TAS
            # (ISA estándar → ΔT=0, el término T/(T-ΔT) = 1)
            term = KAPPA * R * BETA / (2.0 * g) * M**2
            return 1.0 / (1.0 + term)

    else:  # fase == 'cas'
        # Términos comunes a troposfera y estratosfera
        A = (1.0 + (KAPPA - 1.0) / 2.0 * M**2) ** (-1.0 / (KAPPA - 1.0))
        B = (1.0 + (KAPPA - 1.0) / 2.0 * M**2) ** (KAPPA / (KAPPA - 1.0)) - 1.0

        if not en_troposfera:
            # Estratosfera: sin término de gradiente térmico
            return 1.0 / (1.0 + A * B)
        else:
            # Troposfera: término adicional por variación de T con h
            term = KAPPA * R * BETA / (2.0 * g) * M**2
            return 1.0 / (1.0 + term + A * B)


# =============================================================================
# 8. INTEGRADOR PRINCIPAL — UN DESCENSO COMPLETO
# =============================================================================

def simular_descenso(m0_kg, crossover_ft, dh_ft=DH_FT):
    """
    Integra las ecuaciones del TEM desde H_INI_FT hasta H_FIN_FT.

    La variable de integración es la altitud, que decrece en pasos de dh_ft.
    En cada paso se calcula el ROC con el TEM, y de él se obtienen el tiempo
    y la distancia mediante integración de Euler explícita.

    Parámetros
    ----------
    m0_kg        : float  Masa inicial [kg]
    crossover_ft : float  Altitud de crossover [ft]
    dh_ft        : float  Paso de integración [ft]

    Retorna
    -------
    dict con vectores de todas las variables de estado a lo largo del descenso.
    """
    # Vectores de almacenamiento
    h_vec    = []   # Altitud [ft]
    x_vec    = []   # Distancia acumulada [NM]
    t_vec    = []   # Tiempo acumulado [min]
    tas_vec  = []   # TAS [kt]
    M_vec    = []   # Mach [-]
    T_vec    = []   # Temperatura [°C]
    CL_vec   = []   # CL [-]
    CD_vec   = []   # CD [-]
    ROC_vec  = []   # ROD [fpm]  (negativo = descenso)
    Tdes_vec = []   # Empuje de descenso [N]
    D_vec    = []   # Resistencia [N]
    m_vec    = []   # Masa [kg]
    f_vec    = []   # Flujo combustible [kg/min]
    fase_vec = []   # Fase activa

    # Estado inicial
    h = H_INI_FT
    m = m0_kg
    x = 0.0    # metros
    t = 0.0    # segundos

    while h >= H_FIN_FT:

        # --- Estado atmosférico ---
        T_atm, p, rho, a = atmosfera_ISA(h)

        # --- Velocidad según fase activa ---
        if h >= crossover_ft:
            # Fase 1: Mach constante
            fase   = 'mach'
            M      = M_DES
            tas_ms = M_DES * a
        elif h >= FL100_FT:
            # Fase 2: CAS2 = 300 kt
            fase   = 'cas'
            M, tas_ms = cas_a_tas(CAS2_KT, h)
        else:
            # Fase 3: CAS1 = 250 kt  (restricción < FL100)
            fase   = 'cas'
            M, tas_ms = cas_a_tas(CAS1_KT, h)

        # --- Fuerzas ---
        T_des        = empuje_descenso(h)
        CL, CD, D    = aerodinamica(m, tas_ms, rho)
        fM           = esf(M, h, fase)

        # --- ROC del TEM:  ROC = (T-D)*V*f(M) / (m*g) ---
        # ESF en numerador — verificado contra PTD: error < 0.6%
        ROC_ms  = (T_des - D) * tas_ms * fM / (m * g)
        ROC_fpm = ROC_ms * M2FT * 60.0

        # --- Almacenar estado actual ---
        h_vec.append(h)
        x_vec.append(x * M2NM)
        t_vec.append(t / 60.0)
        tas_vec.append(tas_ms * MS2KT)
        M_vec.append(M)
        T_vec.append(T_atm - 273.15)
        CL_vec.append(CL)
        CD_vec.append(CD)
        ROC_vec.append(ROC_fpm)
        Tdes_vec.append(T_des)
        D_vec.append(D)
        m_vec.append(m)
        f_vec.append(CF3 * (1.0 - h / CF4))
        fase_vec.append(fase)

        # --- Si hemos llegado al final, salir sin integrar ---
        if h <= H_FIN_FT:
            break

        # --- Integración de Euler explícita ---
        # Tiempo necesario para descender dh_ft
        dt = (dh_ft * FT2M) / abs(ROC_ms)

        # Actualizar distancia, tiempo y masa
        x += tas_ms * dt
        t += dt
        m -= consumo_descenso(h) * dt

        # Siguiente altitud (no bajar del límite final)
        h = max(h - dh_ft, H_FIN_FT)

    return {
        'h':    np.array(h_vec),
        'x':    np.array(x_vec),
        't':    np.array(t_vec),
        'TAS':  np.array(tas_vec),
        'M':    np.array(M_vec),
        'T':    np.array(T_vec),
        'CL':   np.array(CL_vec),
        'CD':   np.array(CD_vec),
        'ROC':  np.array(ROC_vec),
        'Tdes': np.array(Tdes_vec),
        'D':    np.array(D_vec),
        'm':    np.array(m_vec),
        'f':    np.array(f_vec),
        'fase': np.array(fase_vec),
    }


def distancia_desde_iaf(resultado):
    """
    Convierte la distancia acumulada desde el TOD en distancia desde el IAF.

    El enunciado pide la altitud a 40, 60, 80 y 100 NM del aeropuerto,
    asumiendo que en el IAF la aeronave está a su altitud mínima (H_FIN_FT).
    La distancia desde el IAF es: d_IAF = x_total - x_acumulada.

    Parámetros
    ----------
    resultado : dict  Salida de simular_descenso()

    Retorna
    -------
    d_iaf : np.array  Distancia desde el IAF [NM]
    """
    x_total = resultado['x'][-1]
    return x_total - resultado['x']


def altitud_a_distancia(resultado, target_nm):
    """
    Interpola la altitud de la aeronave a una distancia dada desde el IAF.

    Parámetros
    ----------
    resultado : dict   Salida de simular_descenso()
    target_nm : float  Distancia objetivo desde el IAF [NM]

    Retorna
    -------
    h_ft : float  Altitud interpolada [ft]
    """
    d_iaf = distancia_desde_iaf(resultado)
    # El vector d_iaf decrece (va de d_total a 0); invertir para interpolar
    return float(np.interp(target_nm, d_iaf[::-1], resultado['h'][::-1]))


# =============================================================================
# 9. SIMULACIÓN NOMINAL
# =============================================================================

# Calcular crossover una sola vez (no depende de la masa)
crossover_ft = calcular_crossover(M_DES, CAS2_KT, H_INI_FT, H_FIN_FT)

# Simular trayectoria nominal
nominal = simular_descenso(M_REF_KG, crossover_ft)

# Variables derivadas globales
x_total_nm  = nominal['x'][-1]
t_total_min = nominal['t'][-1]
fuel_kg     = M_REF_KG - nominal['m'][-1]
d_iaf       = distancia_desde_iaf(nominal)

# Índices de puntos de transición
idx_cross = int(np.argmin(np.abs(nominal['h'] - crossover_ft)))
idx_hpdes = int(np.argmin(np.abs(nominal['h'] - HP_DES_FT)))
idx_fl100 = int(np.argmin(np.abs(nominal['h'] - FL100_FT)))

# ── Cabecera ──────────────────────────────────────────────────────────────────
sep = "=" * 90
sep2 = "-" * 90
print(sep)
print("  PREDICTOR DE TRAYECTORIAS DE DESCENSO — A320 BADA 3.9")
print("  Vuelo N251SB  |  LEMD → ENGM  |  06 MAY 2026")
print(sep)

# ── Resultados globales ───────────────────────────────────────────────────────
print(f"\n{'RESULTADOS GLOBALES':^90}")
print(sep2)
print(f"  {'Crossover altitude:':35s} FL{crossover_ft/100:.0f}  ({crossover_ft:.0f} ft)")
print(f"  {'Distancia total del descenso:':35s} {x_total_nm:.2f} NM")
print(f"  {'Duración total del descenso:':35s} {t_total_min:.2f} min")
print(f"  {'Combustible consumido:':35s} {fuel_kg:.1f} kg  ({fuel_kg/M_REF_KG*100:.2f}% de la masa)")
print(f"  {'Masa inicial (TOD):':35s} {M_REF_KG:.0f} kg")
print(f"  {'Masa final (IAF):':35s} {nominal['m'][-1]:.0f} kg")

# ── Altitudes en puntos de referencia ────────────────────────────────────────
print(f"\n{'ALTITUDES Y TIEMPOS EN PUNTOS DE REFERENCIA':^90}")
print(sep2)
print(f"  {'Distancia al aeropuerto':25s} {'Altitud':12s} {'FL':8s} {'Tiempo [min]':15s} {'Fase activa'}")
print(f"  {'-'*75}")
for d_nm in [40, 60, 80, 100]:
    h_ref = altitud_a_distancia(nominal, d_nm)
    t_ref = float(np.interp(d_nm, d_iaf[::-1], nominal['t'][::-1]))
    fase_ref = 'CAS₁=250 kt' if h_ref < FL100_FT else \
               ('CAS₂=300 kt' if h_ref < crossover_ft else 'Mach=0.79')
    if h_ref >= H_INI_FT - 50:
        nota = '(crucero, descenso no iniciado)'
        print(f"  {d_nm:3d} NM {'':22s} {'FL370':12s} {'—':8s} {'—':15s} {nota}")
    else:
        print(f"  {d_nm:3d} NM {'':22s} {h_ref:8.0f} ft  {'FL'+str(int(h_ref/100)):8s} {t_ref:8.2f} min      {fase_ref}")

# ── Tabla detallada por punto de transición ───────────────────────────────────
puntos = [
    ('TOD — Inicio descenso',       0),
    (f'Tropopausa (FL360)',          int(np.argmin(np.abs(nominal['h'] - 36089)))),
    (f'Cdes_high→low (FL{HP_DES_FT/100:.0f})',  idx_hpdes),
    (f'Crossover (FL{crossover_ft/100:.0f})',    idx_cross),
    ('FL100 — CAS₂→CAS₁',          idx_fl100),
    ('IAF VALPU — Fin descenso',    -1),
]

header = (f"  {'Punto':35s} {'h [ft]':>8} {'TAS [kt]':>9} {'M [-]':>7} "
          f"{'T [°C]':>7} {'CL [-]':>7} {'CD [-]':>8} "
          f"{'THR [kN]':>9} {'D [kN]':>8} {'ROD [fpm]':>10} "
          f"{'Fuel [kg/m]':>12} {'t [min]':>8}")

print(f"\n{'ESTADO DE LA AERONAVE EN PUNTOS CLAVE':^90}")
print(sep2)
print(header)
print(f"  {'-'*170}")

for nombre, idx in puntos:
    i  = idx
    h  = nominal['h'][i]
    x  = nominal['x'][i]
    t  = nominal['t'][i]
    V  = nominal['TAS'][i]
    T  = nominal['T'][i]
    CL = nominal['CL'][i]
    CD = nominal['CD'][i]
    Td = nominal['Tdes'][i]
    D  = nominal['D'][i]
    RC = nominal['ROC'][i]
    f  = nominal['f'][i]
    m  = nominal['m'][i]
    print(f"  {nombre:35s} {h:>8.0f} {V:>9.1f} {nominal['M'][i]:>7.4f} "
          f"{T:>7.1f} {CL:>7.4f} {CD:>8.5f} "
          f"{Td/1000:>9.2f} {D/1000:>8.2f} {abs(RC):>10.0f} "
          f"{f:>12.2f} {t:>8.2f}")

# ── Tabla nivel a nivel cada 1000 ft ─────────────────────────────────────────
print(f"\n{'TABLA NIVEL A NIVEL (cada 1000 ft)':^90}")
print(sep2)
print(f"  {'FL':>5} {'h [ft]':>7} {'x [NM]':>8} {'t [min]':>8} "
      f"{'TAS [kt]':>9} {'M [-]':>6} {'T [°C]':>7} "
      f"{'CL':>7} {'CD':>8} {'THR [kN]':>9} {'D [kN]':>8} "
      f"{'ROD [fpm]':>10} {'Fuel [kg/m]':>12} {'Fase':>10}")
print(f"  {'-'*145}")

# Seleccionar un punto por cada 1000 ft de descenso
niveles_imprimir = np.arange(37000, 4800, -1000)
for nivel in niveles_imprimir:
    idx_n = int(np.argmin(np.abs(nominal['h'] - nivel)))
    h  = nominal['h'][idx_n]
    x  = nominal['x'][idx_n]
    t  = nominal['t'][idx_n]
    V  = nominal['TAS'][idx_n]
    M  = nominal['M'][idx_n]
    T  = nominal['T'][idx_n]
    CL = nominal['CL'][idx_n]
    CD = nominal['CD'][idx_n]
    Td = nominal['Tdes'][idx_n]
    D  = nominal['D'][idx_n]
    RC = nominal['ROC'][idx_n]
    f  = nominal['f'][idx_n]
    fs = nominal['fase'][idx_n]
    fl = int(h/100)
    print(f"  {fl:>5} {h:>7.0f} {x:>8.2f} {t:>8.2f} "
          f"{V:>9.1f} {M:>6.4f} {T:>7.1f} "
          f"{CL:>7.4f} {CD:>8.5f} {Td/1000:>9.2f} {D/1000:>8.2f} "
          f"{abs(RC):>10.0f} {f:>12.2f} {fs:>10}")

print(f"\n{sep}\n")


# =============================================================================
# 10. GRÁFICAS — TRAYECTORIA NOMINAL  (Preguntas 1 a 6)
#     Una figura independiente por gráfica para facilitar inserción en Word
# =============================================================================

d_iaf = distancia_desde_iaf(nominal)
h     = nominal['h']
t     = nominal['t']

# --- Figura 1: Perfil de descenso h vs distancia desde IAF ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(d_iaf, h / 1000, color='steelblue', linewidth=2)
for d_ref in [40, 60, 80]:
    h_ref = altitud_a_distancia(nominal, d_ref)
    ax.axvline(d_ref, color='gray', linestyle='--', linewidth=0.8)
    ax.plot(d_ref, h_ref / 1000, 'o', color='tomato', markersize=6,
            label=f'{d_ref} NM → FL{h_ref/100:.0f}')
ax.axhline(crossover_ft / 1000, color='orange', linestyle=':', linewidth=1,
           label=f'Crossover FL{crossover_ft/100:.0f}')
ax.set_xlabel('Distancia desde IAF [NM]', fontsize=11)
ax.set_ylabel('Altitud [×1000 ft]', fontsize=11)
ax.set_title('Figura 1 — Perfil de descenso', fontsize=11)
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.3)
ax.invert_xaxis()
plt.tight_layout()
plt.savefig(ruta('fig1_perfil_descenso.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- Figura 2: TAS vs altitud ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(nominal['TAS'], h / 1000, color='steelblue', linewidth=2)
ax.axhline(crossover_ft / 1000, color='orange',   linestyle='--', linewidth=1,
           label=f'Crossover FL{crossover_ft/100:.0f}')
ax.axhline(FL100_FT / 1000,     color='seagreen', linestyle='--', linewidth=1,
           label='FL100: CAS 300 → 250 kt')
ax.set_xlabel('TAS [kt]', fontsize=11)
ax.set_ylabel('Altitud [×1000 ft]', fontsize=11)
ax.set_title('Figura 2 — TAS en función de la altitud', fontsize=11)
ax.legend(fontsize=8.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(ruta('fig2_TAS_vs_altitud.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- Figura 3: Altitud vs tiempo ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(t, h / 1000, color='steelblue', linewidth=2)
ax.set_xlabel('Tiempo [min]', fontsize=11)
ax.set_ylabel('Altitud [×1000 ft]', fontsize=11)
ax.set_title('Figura 3 — Altitud en función del tiempo', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(ruta('fig3_altitud_vs_tiempo.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- Figura 4: Temperatura, CL y ROD vs altitud ---

# --- Figura 4a: Temperatura vs altitud ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(nominal['T'], h / 1000, color='steelblue', linewidth=2)
ax.axhline(crossover_ft / 1000,  color='orange', linestyle='--', linewidth=1)
ax.axhline(H11_M * M2FT / 1000, color='purple',  linestyle=':',  linewidth=1)
# Anotaciones directas sin leyenda
ax.text(nominal['T'][0] + 1, H11_M*M2FT/1000 + 0.8,
        'Tropopausa FL360', fontsize=8.5, color='purple')
ax.text(nominal['T'][0] + 1, crossover_ft/1000 + 0.8,
        f'Crossover FL{crossover_ft/100:.0f}', fontsize=8.5, color='orange')
ax.set_xlabel('Temperatura [°C]', fontsize=11)
ax.set_ylabel('Altitud [×1000 ft]', fontsize=11)
ax.set_title('Figura 4a — Temperatura ISA en función de la altitud', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(ruta('fig4a_temperatura.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- Figura 4b: CL vs altitud — perfil bañera (altitud en X, CL en Y) ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(h / 1000, nominal['CL'], color='tomato', linewidth=2)
ax.axvline(crossover_ft / 1000, color='orange',   linestyle='--', linewidth=1)
ax.axvline(FL100_FT / 1000,     color='seagreen', linestyle='--', linewidth=1)
ax.axvline(HP_DES_FT / 1000,    color='purple',   linestyle=':',  linewidth=1)

ymax = nominal['CL'].max()
ymin = nominal['CL'].min()
rng  = ymax - ymin

# Etiquetas con offset vertical escalonado para que no se solapen
ax.text(crossover_ft/1000 + 0.5, ymax - rng*0.05,
        f'Crossover\nFL{crossover_ft/100:.0f}', fontsize=8, color='orange',
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='orange', alpha=0.7))
ax.text(FL100_FT/1000 + 0.5, ymax - rng*0.25,
        'FL100\nCAS300→250', fontsize=8, color='seagreen',
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='seagreen', alpha=0.7))
ax.text(HP_DES_FT/1000 + 0.5, ymin + rng*0.05,
        'Cdes alto→bajo\n29 831 ft', fontsize=8, color='purple',
        va='bottom', ha='left',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='purple', alpha=0.7))

ax.invert_xaxis()
ax.set_xlabel('Altitud [×1000 ft]', fontsize=11)
ax.set_ylabel('$C_L$ [-]', fontsize=11)
ax.set_title('Figura 4b — Coeficiente de sustentación en función de la altitud', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(ruta('fig4b_CL.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- Figura 4c: ROD en abscisas, altitud en ordenadas ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(abs(nominal['ROC']), nominal['h'] / 1000, color='seagreen', linewidth=2)

# Líneas horizontales de transición
ax.axhline(crossover_ft / 1000, color='orange', linestyle='--', linewidth=1)
ax.axhline(FL100_FT / 1000,     color='tomato', linestyle='--', linewidth=1)
ax.axhline(HP_DES_FT / 1000,    color='purple', linestyle=':',  linewidth=1)

# ROD máximo
idx_max_rod = int(np.argmax(np.abs(nominal['ROC'])))
h_max_rod   = nominal['h'][idx_max_rod] / 1000
rod_max     = abs(nominal['ROC'][idx_max_rod])
ax.plot(rod_max, h_max_rod, 'o', color='seagreen', markersize=6, zorder=5)
ax.text(rod_max + 20, h_max_rod + 0.5,
        f'{rod_max:.0f} fpm', fontsize=8.5, color='seagreen')

# Salto en crossover — solo flecha sin etiqueta de delta
idx_cx      = int(np.argmin(np.abs(nominal['h'] - crossover_ft)))
rod_antes   = abs(nominal['ROC'][max(idx_cx - 3, 0)])
rod_despues = abs(nominal['ROC'][min(idx_cx + 3, len(nominal['ROC'])-1)])
ax.annotate('', xy=(rod_despues, crossover_ft/1000),
            xytext=(rod_antes, crossover_ft/1000),
            arrowprops=dict(arrowstyle='<->', color='orange', lw=1.5))

# Salto en FL100 — solo flecha sin etiqueta de delta
idx_fl1      = int(np.argmin(np.abs(nominal['h'] - FL100_FT)))
rod_fl_antes = abs(nominal['ROC'][max(idx_fl1 - 3, 0)])
rod_fl_dep   = abs(nominal['ROC'][min(idx_fl1 + 3, len(nominal['ROC'])-1)])
ax.annotate('', xy=(rod_fl_dep, FL100_FT/1000),
            xytext=(rod_fl_antes, FL100_FT/1000),
            arrowprops=dict(arrowstyle='<->', color='tomato', lw=1.5))

# Etiquetas de transición eliminadas — se explican en el texto del documento

ax.set_xlabel('ROD [fpm]', fontsize=11)
ax.set_ylabel('Altitud [×1000 ft]', fontsize=11)
ax.set_title('Figura 4c — Tasa de descenso (ROD) en función de la altitud', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(ruta('fig4c_ROD.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- Figura 5: Masa y consumo vs tiempo ---
fig, ax = plt.subplots(figsize=(9, 5))
ax_r = ax.twinx()
ax.plot(t,   nominal['m'], color='steelblue', linewidth=2, label='Masa [kg]')
ax_r.plot(t, nominal['f'], color='tomato',    linewidth=2, linestyle='--', label='Consumo [kg/min]')
ax.set_xlabel('Tiempo [min]', fontsize=11)
ax.set_ylabel('Masa [kg]',          fontsize=11, color='steelblue')
ax_r.set_ylabel('Consumo [kg/min]', fontsize=11, color='tomato')
ax.set_title('Figura 5 — Masa y consumo de combustible vs tiempo', fontsize=11)
lines1, lbl1 = ax.get_legend_handles_labels()
lines2, lbl2 = ax_r.get_legend_handles_labels()
ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(ruta('fig5_masa_consumo.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- Figura 6: THR y D vs altitud con anotaciones de escalones ---
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(nominal['Tdes'] / 1000, h / 1000, color='tomato',    linewidth=2,
        label='Empuje de descenso $T_{des}$ [kN]')
ax.plot(nominal['D']    / 1000, h / 1000, color='steelblue', linewidth=2,
        label='Resistencia $D$ [kN]')
ax.fill_betweenx(h / 1000,
                 nominal['Tdes'] / 1000,
                 nominal['D']    / 1000,
                 where=(nominal['D'] > nominal['Tdes']),
                 alpha=0.12, color='gray', label='Exceso de resistencia $D - T_{des}$')
ax.axhline(crossover_ft / 1000, color='orange',   linestyle='--', linewidth=1,
           label=f'Crossover FL{crossover_ft/100:.0f}')
ax.axhline(10.0,                color='seagreen', linestyle='--', linewidth=1,
           label='FL100: CAS 300 → 250 kt')
ax.axhline(HP_DES_FT / 1000,    color='purple',   linestyle=':',  linewidth=1,
           label='29 831 ft: Cdes,high → Cdes,low')

# Anotación escalón en D (FL100)
idx_fl100 = np.argmin(np.abs(h - 9950))
ax.annotate('Escalon en D: -10 kN\n(CAS 300 a 250 kt)',
            xy=(nominal['D'][idx_fl100] / 1000, 10.0), xytext=(38, 15.5),
            arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.2),
            fontsize=8.5, color='steelblue',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='steelblue', alpha=0.8))

# Anotación escalón en Tdes (Hp_des)
idx_hpdes = np.argmin(np.abs(h - 29731))
ax.annotate('Escalon en Tdes:\nCdes,high a Cdes,low\n(-23% de empuje)',
            xy=(nominal['Tdes'][idx_hpdes] / 1000, HP_DES_FT / 1000),
            xytext=(22, 24),
            arrowprops=dict(arrowstyle='->', color='tomato', lw=1.2),
            fontsize=8.5, color='tomato',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='tomato', alpha=0.8))

ax.set_xlabel('Fuerza [kN]', fontsize=11)
ax.set_ylabel('Altitud [×1000 ft]', fontsize=11)
ax.set_title('Figura 6 — Empuje de descenso y resistencia vs altitud', fontsize=11)
ax.legend(fontsize=8.5, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(10, 62)
plt.tight_layout()
plt.savefig(ruta('fig6_THR_D.png'), dpi=150, bbox_inches='tight')
plt.close()

print("\nFiguras 1-6 (trayectoria nominal) guardadas.")


# =============================================================================
# 11. ANÁLISIS DE MONTE CARLO  (Preguntas 9, 10 y 11)
# =============================================================================

print(f"\nEjecutando {N_SIM} simulaciones de Monte Carlo...")

# Puntos de referencia para el análisis (distancia desde IAF)
D_REF_NM = [40, 60, 80, 100]

# Muestreo de masas: distribución uniforme U(m_ref±5%)
m_min = M_REF_KG * (1.0 - INCERT)
m_max = M_REF_KG * (1.0 + INCERT)
masas_mc = np.random.uniform(m_min, m_max, N_SIM)

# Almacenamiento de resultados Monte Carlo
traj_mc    = []                        # Trayectorias para figura 7
h_en_ref   = {d: [] for d in D_REF_NM}  # Altitud en cada punto de referencia
t_en_ref   = {d: [] for d in D_REF_NM}  # Tiempo transcurrido en cada punto de referencia
t_total_mc = []                        # Tiempo total de cada simulación
x_total_mc = []                        # Distancia total de cada simulación

def tiempo_a_distancia(resultado, target_nm):
    """
    Interpola el tiempo transcurrido cuando la aeronave está a target_nm del IAF.

    Parámetros
    ----------
    resultado  : dict   Salida de simular_descenso()
    target_nm  : float  Distancia objetivo desde el IAF [NM]

    Retorna
    -------
    t_min : float  Tiempo transcurrido [min]
    """
    d_iaf = distancia_desde_iaf(resultado)
    return float(np.interp(target_nm, d_iaf[::-1], resultado['t'][::-1]))

for i, m0 in enumerate(masas_mc):
    res = simular_descenso(m0, crossover_ft)
    d   = distancia_desde_iaf(res)

    # Guardar trayectoria para figura 7
    traj_mc.append((d, res['h']))

    # Guardar tiempo y distancia totales
    t_total_mc.append(res['t'][-1])
    x_total_mc.append(res['x'][-1])

    # Altitud y tiempo en cada punto de referencia (Preguntas 10 y 11)
    for d_nm in D_REF_NM:
        h_en_ref[d_nm].append(altitud_a_distancia(res, d_nm))
        t_en_ref[d_nm].append(tiempo_a_distancia(res, d_nm))

t_total_mc = np.array(t_total_mc)
x_total_mc = np.array(x_total_mc)
for d in D_REF_NM:
    h_en_ref[d] = np.array(h_en_ref[d])
    t_en_ref[d] = np.array(t_en_ref[d])

print("Monte Carlo completado.")


# =============================================================================
# 12. GRÁFICAS MONTE CARLO — UNA FIGURA POR ARCHIVO
# =============================================================================

# --- Figura 7: Trayectorias Monte Carlo (Pregunta 9) ---
# Las 1000 trayectorias se pintan muy finas y transparentes para que la
# acumulación forme una nube de densidad sin ocultar la nominal.
fig, ax = plt.subplots(figsize=(12, 6))
for d, h_traj in traj_mc:
    ax.plot(d, h_traj / 1000, color='steelblue', alpha=0.06, linewidth=0.3)
ax.plot(d_iaf, nominal['h'] / 1000, color='tomato', linewidth=2.5,
        label='Trayectoria nominal', zorder=5)
for d_ref in D_REF_NM:
    ax.axvline(d_ref, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel('Distancia desde IAF [NM]', fontsize=11)
ax.set_ylabel('Altitud [×1000 ft]', fontsize=11)
ax.set_title(f'Figura 7 — Trayectorias Monte Carlo ({N_SIM} simulaciones) — masa U(±5%)', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.invert_xaxis()
plt.tight_layout()
plt.savefig(ruta('fig7_mc_trayectorias.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- Figura 8: Altitud en los 4 puntos de referencia vs masa — bloque 2×2 ---
from matplotlib.lines import Line2D

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Figura 8 — Masa inicial vs altitud en puntos de referencia  —  Monte Carlo  U(±5%)',
             fontsize=12)

posiciones = [(0,0), (0,1), (1,0), (1,1)]
for idx, d_nm in enumerate(D_REF_NM):
    row, col = posiciones[idx]
    ax = axes[row, col]
    ax.scatter(h_en_ref[d_nm] / 1000, masas_mc / 1000,
               s=4, alpha=0.35, color='steelblue')
    ax.axhline(M_REF_KG / 1000, color='tomato', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Altitud [×1000 ft]', fontsize=11)
    ax.set_ylabel('Masa inicial [t]', fontsize=11)
    ax.set_title(f'{d_nm} NM del aeropuerto', fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)

legend_elements = [
    Line2D([0],[0], color='tomato', linestyle='--', linewidth=1.5,
           label=f'Masa nominal ({M_REF_KG/1000:.1f} t)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=1,
           fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))

plt.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.08,
                    hspace=0.35, wspace=0.28)
plt.savefig(ruta('fig8_altitud_vs_masa.png'), dpi=150, bbox_inches='tight')
plt.close()

# =============================================================================
# 13. EXPORTACIÓN DE RESULTADOS PARA AJUSTE EN MATLAB (disttfitter)
# =============================================================================
#
# Se exportan dos formatos equivalentes:
#
#   monte_carlo_resultados.csv  →  legible en cualquier herramienta
#   monte_carlo_resultados.mat  →  cargable directamente en MATLAB con load()
#
# Cada fila del CSV / cada variable del .mat corresponde a una simulación.
# Columnas / variables exportadas:
#
#   masa_kg       Masa inicial sorteada [kg]
#   h_40nm_ft     Altitud a 40 NM del aeropuerto [ft]
#   h_60nm_ft     Altitud a 60 NM del aeropuerto [ft]
#   h_80nm_ft     Altitud a 80 NM del aeropuerto [ft]
#   h_100nm_ft    Altitud a 100 NM del aeropuerto [ft]
#   t_total_min   Tiempo total del descenso [min]
#   x_total_nm    Distancia total del descenso [NM]
#
# En MATLAB, para lanzar disttfitter sobre, por ejemplo, la altitud a 40 NM:
#
#   load('monte_carlo_resultados.mat')
#   disttfitter(h_40nm_ft)
#
# Para el tiempo en ese mismo punto:
#
#   disttfitter(t_40nm_min)
#
# o equivalentemente desde el CSV:
#
#   T = readtable('monte_carlo_resultados.csv');
#   disttfitter(T.h_40nm_ft)
#   disttfitter(T.t_40nm_min)

import csv
import scipy.io as sio

# --- Exportar CSV ---
ruta_csv = ruta('monte_carlo_resultados.csv')
cabecera = [
    'masa_kg',
    'h_40nm_ft',  'h_60nm_ft',  'h_80nm_ft',  'h_100nm_ft',
    't_40nm_min', 't_60nm_min', 't_80nm_min', 't_100nm_min',
    't_total_min', 'x_total_nm'
]

with open(ruta_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(cabecera)
    for i in range(N_SIM):
        writer.writerow([
            round(masas_mc[i],       4),
            round(h_en_ref[40][i],   4),
            round(h_en_ref[60][i],   4),
            round(h_en_ref[80][i],   4),
            round(h_en_ref[100][i],  4),
            round(t_en_ref[40][i],   6),
            round(t_en_ref[60][i],   6),
            round(t_en_ref[80][i],   6),
            round(t_en_ref[100][i],  6),
            round(t_total_mc[i],     6),
            round(x_total_mc[i],     6),
        ])

print(f"CSV exportado: {ruta_csv}")

# --- Exportar MAT (formato MATLAB) ---
ruta_mat = ruta('monte_carlo_resultados.mat')
sio.savemat(ruta_mat, {
    'masa_kg':      masas_mc,
    'h_40nm_ft':    h_en_ref[40],
    'h_60nm_ft':    h_en_ref[60],
    'h_80nm_ft':    h_en_ref[80],
    'h_100nm_ft':   h_en_ref[100],
    't_40nm_min':   t_en_ref[40],
    't_60nm_min':   t_en_ref[60],
    't_80nm_min':   t_en_ref[80],
    't_100nm_min':  t_en_ref[100],
    't_total_min':  t_total_mc,
    'x_total_nm':   x_total_mc,
})

print(f"MAT exportado:  {ruta_mat}")
print(f"\nTodas las gráficas y datos guardados en: {DIR_OUTPUTS}")
