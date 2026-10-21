# 1. Librerías y configuración reproducible
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from IPython.display import clear_output
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

SEMILLA = 42
random.seed(SEMILLA)
np.random.seed(SEMILLA)
tf.random.set_seed(SEMILLA)
tf.keras.backend.clear_session()

print('TensorFlow:', tf.__version__)
print('Configuración lista.')

# 2. Dataset completo
X, y = make_moons(n_samples=1200, noise=0.28, random_state=SEMILLA)

plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=24, alpha=0.8)
plt.title('Dataset completo: el color representa la clase')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.grid(alpha=0.2)
plt.show()

# 3. División 60 % / 20 % / 20 %
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, stratify=y, random_state=SEMILLA
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEMILLA
)

# El escalador aprende SOLO con entrenamiento
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True, sharey=True)
for ax, datos, etiquetas, titulo in [
    (axes[0], X_train_s, y_train, f'Entrenamiento ({len(y_train)})'),
    (axes[1], X_val_s, y_val, f'Validación ({len(y_val)})'),
    (axes[2], X_test_s, y_test, f'Prueba: nunca vistos ({len(y_test)})')
]:
    ax.scatter(datos[:, 0], datos[:, 1], c=etiquetas, cmap='coolwarm', s=22, alpha=0.8)
    ax.set_title(titulo)
    ax.set_xlabel('$x_1$ estandarizada')
    ax.grid(alpha=0.2)
axes[0].set_ylabel('$x_2$ estandarizada')
plt.tight_layout()
plt.show()

# 4. Visualización de una frontera de decisión
def dibujar_frontera(modelo, X_datos, y_datos, titulo, ax=None, paso=0.04):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    margen = 0.8
    x_min, x_max = X_datos[:, 0].min() - margen, X_datos[:, 0].max() + margen
    y_min, y_max = X_datos[:, 1].min() - margen, X_datos[:, 1].max() + margen
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, paso),
        np.arange(y_min, y_max, paso)
    )
    rejilla = np.c_[xx.ravel(), yy.ravel()]
    prob = modelo.predict(rejilla, verbose=0).reshape(xx.shape)

    ax.contourf(xx, yy, prob, levels=np.linspace(0, 1, 21), cmap='coolwarm', alpha=0.45)
    ax.contour(xx, yy, prob, levels=[0.5], colors='black', linewidths=2)
    ax.scatter(X_datos[:, 0], X_datos[:, 1], c=y_datos, cmap='coolwarm',
               edgecolor='white', linewidth=0.4, s=28)
    ax.set_title(titulo)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.grid(alpha=0.15)
    return ax

# 5. Callback visual corregido
class EntrenamientoVisual(tf.keras.callbacks.Callback):
    def __init__(self, X_mostrar, y_mostrar, cada=5, nombre='Modelo'):
        super().__init__()
        self.X_mostrar = X_mostrar
        self.y_mostrar = y_mostrar
        self.cada = cada
        self.nombre = nombre

        # Cada historial se crea por separado para evitar asignaciones ambiguas.
        self.epocas = []
        self.loss = []
        self.val_loss = []
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epocas.append(epoch + 1)
        self.loss.append(logs.get('loss', np.nan))
        self.val_loss.append(logs.get('val_loss', np.nan))
        self.acc.append(logs.get('accuracy', np.nan))
        self.val_acc.append(logs.get('val_accuracy', np.nan))

        mostrar = ((epoch + 1) % self.cada == 0) or (epoch == 0)
        if mostrar:
            clear_output(wait=True)
            fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

            axes[0].plot(self.epocas, self.loss, label='Entrenamiento', lw=2)
            axes[0].plot(self.epocas, self.val_loss, label='Validación', lw=2)
            axes[0].set(title='Pérdida', xlabel='Época', ylabel='Binary cross-entropy')
            axes[0].legend()
            axes[0].grid(alpha=0.25)

            axes[1].plot(self.epocas, self.acc, label='Entrenamiento', lw=2)
            axes[1].plot(self.epocas, self.val_acc, label='Validación', lw=2)
            axes[1].set(title='Exactitud', xlabel='Época', ylabel='Accuracy', ylim=(0, 1.03))
            axes[1].legend()
            axes[1].grid(alpha=0.25)

            dibujar_frontera(
                self.model, self.X_mostrar, self.y_mostrar,
                f'{self.nombre} — época {epoch + 1}', ax=axes[2]
            )
            plt.tight_layout()
            plt.show()


# 6. Crear una red base
tf.keras.backend.clear_session()
tf.random.set_seed(SEMILLA)

modelo_base = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
], name='red_base')

modelo_base.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

modelo_base.summary()

# 7. Entrenar y visualizar cada 5 épocas
visual_base = EntrenamientoVisual(
    X_mostrar=X_val_s,
    y_mostrar=y_val,
    cada=5,
    nombre='A · Red base'
)

hist_base = modelo_base.fit(
    X_train_s,
    y_train,
    validation_data=(X_val_s, y_val),
    epochs=80,
    batch_size=32,
    verbose=0,
    callbacks=[visual_base]
)

print('Entrenamiento terminado.')
# 8. Comparar los tres subconjuntos
resultados_base = []
for nombre, X_parte, y_parte in [
    ('Entrenamiento', X_train_s, y_train),
    ('Validación', X_val_s, y_val),
    ('Prueba (nunca vistos)', X_test_s, y_test)
]:
    perdida, exactitud = modelo_base.evaluate(X_parte, y_parte, verbose=0)
    resultados_base.append([nombre, perdida, exactitud])

pd.DataFrame(resultados_base, columns=['Datos', 'Pérdida', 'Exactitud']).round(4)



# 9. Ver predicciones individuales de prueba
prob_test = modelo_base.predict(X_test_s, verbose=0).ravel()
pred_test = (prob_test >= 0.5).astype(int)

indices = np.arange(12)
tabla_pred = pd.DataFrame({
    'x1': X_test_s[indices, 0],
    'x2': X_test_s[indices, 1],
    'Clase real': y_test[indices],
    'Probabilidad clase 1': prob_test[indices],
    'Predicción': pred_test[indices],
    '¿Correcta?': pred_test[indices] == y_test[indices]
})
tabla_pred.round(3)

# 10. Matriz de confusión y frontera final sobre prueba
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
dibuja = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, pred_test),
    display_labels=['Clase 0', 'Clase 1']
)
dibuja.plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title('Matriz de confusión — prueba')
dibuja_frontera = dibujar_frontera(
    modelo_base, X_test_s, y_test,
    'Frontera final sobre datos nunca vistos', ax=axes[1]
)
plt.tight_layout()
plt.show()

# 11. Conjunto pequeño para provocar sobreajuste
rng = np.random.default_rng(SEMILLA)
idx_pequeno = rng.choice(len(X_train_s), size=200, replace=False)
X_train_peq = X_train_s[idx_pequeno]
y_train_peq = y_train[idx_pequeno]

plt.figure(figsize=(6, 4.5))
plt.scatter(X_train_peq[:, 0], X_train_peq[:, 1], c=y_train_peq,
            cmap='coolwarm', edgecolor='white', s=45)
plt.title('Solo 80 ejemplos para una red grande')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.grid(alpha=0.2)
plt.show()


# 12. Dos modelos comparables: sin regularización y con Dropout
def crear_modelo_grande(dropout=0.0, semilla=42):
    tf.random.set_seed(semilla)
    capas = [tf.keras.layers.Input(shape=(2,))]
    for _ in range(4):
        capas.append(tf.keras.layers.Dense(128, activation='relu'))
        if dropout > 0:
            capas.append(tf.keras.layers.Dropout(dropout))
    capas.append(tf.keras.layers.Dense(1, activation='sigmoid'))

    modelo = tf.keras.Sequential(capas)
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return modelo

tf.keras.backend.clear_session()
modelo_overfit = crear_modelo_grande(dropout=0.0, semilla=SEMILLA)
modelo_dropout = crear_modelo_grande(dropout=0.10, semilla=SEMILLA)

print('Parámetros sin regularización:', modelo_overfit.count_params())
print('Parámetros con Dropout:', modelo_dropout.count_params())
print('Dropout no agrega parámetros: apaga neuronas aleatoriamente durante entrenamiento.')

# 13. Entrenar ambos modelos bajo las mismas condiciones
EPOCAS_COMPARACION = 180

hist_overfit = modelo_overfit.fit(
    X_train_peq, y_train_peq,
    validation_data=(X_val_s, y_val),
    epochs=EPOCAS_COMPARACION,
    batch_size=16,
    verbose=0
)

hist_dropout = modelo_dropout.fit(
    X_train_peq, y_train_peq,
    validation_data=(X_val_s, y_val),
    epochs=EPOCAS_COMPARACION,
    batch_size=16,
    verbose=0
)

print('Comparación terminada.')

# 14. Curvas: memorizar vs generalizar
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

axes[0, 0].plot(hist_overfit.history['loss'], label='Entrenamiento')
axes[0, 0].plot(hist_overfit.history['val_loss'], label='Validación')
axes[0, 0].set_title('Sin regularización — pérdida')
axes[0, 0].set_ylabel('Binary cross-entropy')

axes[0, 1].plot(hist_dropout.history['loss'], label='Entrenamiento')
axes[0, 1].plot(hist_dropout.history['val_loss'], label='Validación')
axes[0, 1].set_title('Con Dropout (0.40) — pérdida')

axes[1, 0].plot(hist_overfit.history['accuracy'], label='Entrenamiento')
axes[1, 0].plot(hist_overfit.history['val_accuracy'], label='Validación')
axes[1, 0].set_title('Sin regularización — exactitud')
axes[1, 0].set_ylabel('Accuracy')

axes[1, 1].plot(hist_dropout.history['accuracy'], label='Entrenamiento')
axes[1, 1].plot(hist_dropout.history['val_accuracy'], label='Validación')
axes[1, 1].set_title('Con Dropout (0.40) — exactitud')

for ax in axes.ravel():
    ax.set_xlabel('Época')
    ax.grid(alpha=0.25)
    ax.legend()

plt.tight_layout()
plt.show()
