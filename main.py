import math
from sqlite3 import Time

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt


def generate_synthetic_data(
    start_date,
    end_date,
    freq="1min",
    seed=None,
    p_mean=5.0,
    p_std=0.1,
    t_mean=70.0,
    t_std=0.5,
):
    rng = pd.date_range(start=start_date, end=end_date, freq=freq)
    n = len(rng)
    np.random.seed(seed)

    data = {
        "timestamp": rng,
        "sensor_id": "SENSOR-A",
        # loc = média, scale = desvio padrão (volatilidade)
        "pressure": np.random.normal(loc=p_mean, scale=p_std, size=n),
        "temperature": np.random.normal(loc=t_mean, scale=t_std, size=n),
    }
    return pd.DataFrame(data).head(n // 2)


# Gerar dados sintéticos para 7 dias
df = generate_synthetic_data(
    "2024-01-01",
    "2024-01-08",
    seed=42,
    t_std=0.0001,
    p_std=0.0001,
    t_mean=500,
    p_mean=500,
)

# Paginação dos dados gerados
pag_tam = 10
pag = 0
strt = pag * pag_tam
end = strt + pag_tam
rept = True
print("=====================================")
print(f"\nTotal de Linhas geradas: {len(df):,}")
while rept:
    print(df.iloc[strt:end])
    print(f"Página {pag + 1} de {len(df) // pag_tam + 1}")
    choice = input(
        "                                      Digite 1 para avançar, 2 para voltar, ou Enter para sair: \n"
    )
    if choice == "1":
        if end >= len(df):
            print("Você já está na última página.")
            continue
        pag += 1
        strt = pag * pag_tam
        end = strt + pag_tam
    elif choice == "2":
        if pag == 0:
            print("Você já está na primeira página.")
            continue
        pag -= 1
        strt = pag * pag_tam
        end = strt + pag_tam
    else:
        print("Encerrando a visualização.")
        rept = False

# Estilo do Grafico
sns.set_style("whitegrid")

# Pegando os valores da amostra
x = df["temperature"]  # Variável independente (temperatura)
y = df["pressure"]  # Variável dependente (pressão)

# Calculando os coeficientes e intercepto da linha de regressão
coef, intercept = np.polyfit(x, y, 1)
print(f"Coeficiente angular (slope): {coef:.4f}")
print(f"Intercepto (intercept): {intercept:.4f}")

# Disperção + Linha de tendência
formula = rf"y = {coef:.1g}x + {intercept:.2f}"
sns.regplot(
    data=df,
    x="temperature",
    y="pressure",
    truncate=False,
    scatter_kws={
        "alpha": 0.5,
        "color": "purple",
        "label": "Data Points",
        "zorder": 1,
        "s": 50,
        "marker": "o",
    },
    line_kws={
        "color": "pink",
        "label": formula,
        "linestyle": "-",
        "linewidth": 2,
        "zorder": 2,
    },
)

# Bunitezas

plt.legend(title="Legenda", fontsize=10)
plt.title("Relação entre Pressão e Temperatura", fontsize=16)
plt.xlabel("Temperatura (°F)", fontsize=12)
plt.ylabel("Pressão (bar)", fontsize=12)
plt.gca().set_aspect("equal", adjustable="datalim", anchor="C")

# Exibir o gráfico
plt.show()
