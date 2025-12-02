# 👁️ ROG Vision - Sistema de Reconocimiento Facial

Este proyecto es un sistema avanzado de autenticación biométrica diseñado para identificar usuarios específicos ("Alex" y "Oscar") en tiempo real utilizando Inteligencia Artificial.

## 🚀 Características

*   **Detección en Vivo:** Identificación instantánea mediante webcam.
*   **Modelo IA Avanzado:** Utiliza **MobileNetV2** (Transfer Learning) para alta precisión.
*   **Anti-Spoofing Básico:** Filtros de visión (CLAHE, GaussianBlur) y umbrales estrictos para evitar falsos positivos.
*   **Arquitectura Moderna:**
    *   **Backend:** FastAPI (Python) para inferencia rápida.
    *   **Frontend:** Streamlit para una interfaz visual atractiva y reactiva.
    *   **Contenedores:** Dockerizado para fácil despliegue.

## 🛠️ Tecnologías

*   **Python 3.9**
*   **TensorFlow / Keras**
*   **OpenCV**
*   **FastAPI**
*   **Streamlit**
*   **Docker**

## 📂 Estructura del Proyecto

```
ProjectVision/
├── backend/            # API de Inferencia (FastAPI)
├── frontend_st/        # Interfaz de Usuario (Streamlit)
├── modules/
│   ├── data_collection/ # Scripts de captura y procesamiento de dataset
│   ├── training/        # Scripts de entrenamiento del modelo
│   └── models/          # Modelos entrenados (.h5) y metadatos
├── start.ps1           # Script de inicio rápido (Windows)
└── requirements.txt    # Dependencias del proyecto
```

## ⚡ Guía de Inicio Rápido

### 1. Requisitos Previos
*   Docker Desktop instalado y corriendo.
*   Python 3.9+ (para scripts locales).
*   Webcam funcional.

### 2. Instalación
Clona este repositorio:
```bash
git clone https://github.com/GsDesk/ProjectVision.git
cd ProjectVision
```

### 3. Ejecución
Simplemente ejecuta el script de inicio en PowerShell:
```powershell
./start.ps1
```
Esto levantará automáticamente los servicios de Backend y Frontend.
*   **Frontend:** [http://localhost:8501](http://localhost:8501)
*   **Backend Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

## 🧠 Entrenamiento del Modelo (Opcional)

Si deseas agregar nuevas caras o re-entrenar:

1.  **Captura de Datos:**
    ```bash
    python modules/data_collection/capture.py Alex
    ```
2.  **Procesamiento (Recorte de Caras):**
    ```bash
    python modules/data_collection/process_dataset.py
    ```
3.  **Entrenamiento:**
    ```bash
    python modules/training/train.py
    ```
4.  **Reiniciar Sistema:**
    ```powershell
    ./start.ps1
    ```

## 📝 Notas
*   El sistema está configurado para distinguir entre **Alex** y **Oscar**.
*   Cualquier otra persona será clasificada como **"Desconocido"** si la confianza es menor al 92%.

---
Desarrollado por **GsDesk**
