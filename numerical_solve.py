import numpy as np
from scipy.optimize import fsolve

# Constantes de la tabla
q = 1.602176634e-19  # Carga elemental en C
k = 1.3880649e-23  # Constante de Boltzmann en J/K
T = 300  # Temperatura en K
D_n = 2  # Coeficiente de difusión de electrones en cm^2/s
D_p = 11.6  # Coeficiente de difusión de huecos en cm^2/s
tau_n = 371e-6  # Tiempo de vida de electrones en s
tau_p = 3710e-6  # Tiempo de vida de huecos en s
epsilon = 1.035918e-12  # Permitividad eléctrica del silicio en F/cm
n_i = 9.69600e9  # Concentración intrínseca de portadores en cm^-3
J_L = 50e-3 


N_A = 1e16  # Concentración de aceptores en cm^-3
N_D = 5e20  # Concentración de donadores en cm^-3
W_n = 0.0000750  # Ancho de la región n en cm
W_p = 0.02  # Ancho de la región p en cm


# Definir la función coth(x)
def coth(x):
    return np.cosh(x) / np.sinh(x) if np.sinh(x) != 0 else 1

# Definir la función para calcular V_int
def calculate_V_int(N_A, N_D):
    return (k * T / q) * np.log(N_A * N_D / n_i**2)

# Definir la función para calcular X_n y X_p
def calculate_X_n(V_int, epsilon, q, N_A, N_D):
    return np.sqrt(2 * epsilon * V_int / q * (N_A / (N_D * (N_A + N_D))))

def calculate_X_p(V_int, epsilon, q, N_A, N_D):
    return np.sqrt(2 * epsilon * V_int / q * (N_D / (N_A * (N_A + N_D))))

# Actualizar la función para calcular J_D con el término n_i^2 correctamente incorporado
def calculate_J_D(V, W_n, W_p, N_D, N_A):
    # Calcular V_int, X_n, X_p para los parámetros dados
    V_int = calculate_V_int(N_A, N_D)
    X_n = calculate_X_n(V_int, epsilon, q, N_A, N_D)
    X_p = calculate_X_p(V_int, epsilon, q, N_A, N_D)

    # Termino exponencial con n_i^2
    exponential_term = (np.exp(q * V / (k * T)) - 1) * n_i**2

    # Termino 1: para electrones
    D_n = 2  # Coeficiente de difusión de electrones en cm^2/s
    tau_n = 371e-6  # Tiempo de vida de electrones en s
    term_1 = np.sqrt(D_n / tau_n) / N_A * coth((X_p + W_n) / np.sqrt(D_n * tau_n))

    # Termino 2: para huecos
    D_p = 11.6  # Coeficiente de difusión de huecos en cm^2/s
    tau_p = 3710e-6  # Tiempo de vida de huecos en s
    term_2 = np.sqrt(D_p / tau_p) / N_D * coth((X_n + W_p) / np.sqrt(D_p * tau_p))

    # Calcular J_D con n_i^2 correctamente incorporado
    J_D = q * exponential_term * (term_1 + term_2)

    return J_D

# Definir la función de potencia
def calculate_power(params):
    V, W_n, W_p, N_D, N_A = params
    # Calcular J_D para los parámetros actuales
    J_D = calculate_J_D(V, W_n, W_p, N_D, N_A)
    # Calcular la corriente neta
    J = J_L - J_D
    # Calcular la potencia
    P = J * V
    # Queremos maximizar la potencia, por lo que devolvemos -P para minimizar
    return -P

def equation(V):
    J_D = calculate_J_D(V, W_n, W_p, N_D, N_A)
    term1 = 1 + (J_L *  (np.exp((q*V)/(k*T))-1)/ J_D)
    term2 = np.exp((q * V) / (k * T)) * ((q * V) / (k * T) + 1)
    return term1 - term2

# Usar fsolve para encontrar el valor de V
V_initial_guess = 4  # Una suposición inicial para V
V_solution = fsolve(equation, V_initial_guess)

print(f"V Solution is: {V_solution}")