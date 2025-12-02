Write-Host " Iniciando Sistema de Reconocimiento Facial..." -ForegroundColor Yellow

# Detener lo anterior
docker stop api-vision mi-interfaz 2>$null
docker rm api-vision mi-interfaz 2>$null

# Iniciar Backend
Write-Host " Cargando Cerebro (Backend)..."
docker run -d -p 8000:8000 --name api-vision -v ${PWD}/backend/app:/app/app -v ${PWD}/modules:/app/modules vision-facial-app

# Iniciar Frontend
Write-Host " Cargando Interfaz (Frontend)..."
docker run -d -p 8501:8501 --add-host host.docker.internal:host-gateway --name mi-interfaz -v ${PWD}/frontend_st/app.py:/app/app.py frontend-vision

Write-Host " ¡TODO LISTO!" -ForegroundColor Green
Write-Host " Ve a: http://localhost:8501" -ForegroundColor Cyan
