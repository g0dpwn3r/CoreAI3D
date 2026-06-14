# CoreAI3D Commercialization Plan

## Fase 1: Secure Authentication Foundation

### Stap 1.1: Auth API Endpoint toevoegen aan C++ Backend
**Bestand**: `/CoreAI3D/CoreAI3D/src/APIServer.cpp`
- Voeg `/auth/login` endpoint toe (username/password → JWT)
- Gebruik bcrypt voor wachtwoord hashing (al beschikbaar in vcpkg)
- JWT library toevoegen: `nlohmann/json` voor token payload
- Session management via MySQL database

### Stap 1.2: Database schema uitbreiden
**Bestand**: Nieuw bestand `/CoreAI3D/CoreAI3D/src/AuthTables.cpp`
```sql
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE sessions (
    id VARCHAR(128) PRIMARY KEY,
    user_id INT REFERENCES users(id),
    jwt_token TEXT,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE api_keys (
    key VARCHAR(64) PRIMARY KEY,
    user_id INT REFERENCES users(id),
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Stap 1.3: Python GUI Auth integratie
**Bestand**: `/CoreAI3D/linux_gui/coreai3d_client.py`
- Voeg `login(username, password)` methode toe
- Sla JWT op in client config
- Voeg token validatie toe voor elke API call

## Fase 2: HuggingFace Training Pipeline

### Stap 2.1: Training module API uitbreiden
**Bestand**: `/CoreAI3D/CoreAI3D/include/Train.hpp` en `src/Train.cpp`
- Voeg HuggingFace model download endpoint toe: `/training/load_model`
- Voeg model-specific training toe: `/training/finetune`
- Asynchronous processing met Celery-style queue in C++

### Stap 2.2: Python training helper
**Bestand**: Nieuw `/CoreAI3D/python/training_hf.py`
```python
# Async HuggingFace training wrapper
# Download base models via HF API
# Fine-tune met CoreAI3D datasets
# Push naar private HF repo
```

### Stap 2.3: Training status WebSocket
- Voeg training progress events toe aan WebSocket stream
- Real-time metrics: loss, accuracy, epoch progress

## Fase 3: Commerciële Website

### Stap 3.1: Next.js Frontend structuur
**Directory**: `/CoreAI3D/commercial_web/`
```
/app/
  /api/
    /auth/login/route.js
    /training/route.js
    /prediction/route.js
  /dashboard/
    page.js (training monitoring)
  /models/
    page.js (HuggingFace model discovery)
/components/
  ModelTopology3D.jsx (Three.js visualisatie)
  TrainingDashboard.jsx
  ChatInterface.jsx
```

### Stap 3.2: API proxy routes
- `/api/*` routes die authentiseren en proxy naar CoreAI3D backend
- Rate limiting per gebruiker
- Stripe webhook integratie voor abonnementen

### Stap 3.3: 3D Neural Network Visualisatie
**Bibliotheken**: Three.js, React Three Fiber
- Visualiseer model architecture
- Live training metrics in 3D ruimte
- Transparantie: toon exact data flow

## Fase 4: Container Orchestratie

### Stap 4.1: Productie Docker image
**Bestand**: `/CoreAI3D/commercial_docker/Dockerfile`
- Separere auth API container
- CoreAI3D training container
- Redis voor job queue tussen containers

### Stap 4.2: Docker Compose setup
```yaml
version: '3.8'
services:
  coreai3d-api:
    build: ../
    ports:
      - "8080:8080"
      - "8081:8081"
  redis:
    image: redis:alpine
  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: ${MYSQL_ROOT_PASSWORD}
```

## Fase 5: Nuitka Executable Build

### Stap 5.1: GUI klaar maken voor compilatie
**Bestand**: `/CoreAI3D/linux_gui/run_dashboard.py`
- Verwijder qasync event loop blocking (compatibiliteit met Nuitka)
- Gebruik pure asyncio of threading
- Test cross-platform compatibiliteit

### Stap 5.2: Build script
**Bestand**: `/CoreAI3D/build_executable.sh`
```bash
#!/bin/bash
python3 -m nuitka \
  --follow-imports \
  --standalone \
  --onefile \
  --output-dir=dist \
  linux_gui/dashboard.py
```

## Technische Afhankelijkheden Nieuw

### C++ Libraries (via vcpkg)
- `jwt-cpp` - JWT token handling
- `bcrypt` - Wachtwoord hashing
- `torch` - HuggingFace model inference
- `huggingface-hub-cpp` (of eigen wrapper)

### Web Libraries
- `next`, `react`, `react-dom`
- `@stripe/stripe-js`
- `three`, `@react-three/fiber`
- `socket.io-client`

### Python Libraries (aanvullende)
- `transformers`
- `datasets`
- `peft` (Parameter-Efficient Fine-Tuning)
- `accelerate`

## Security Checklist

- [ ] Wachtwoorden met bcrypt + salt
- [ ] JWT tokens met 15 min expiry + refresh token flow
- [ ] HTTPS enforcement (Let's Encrypt certificaten)
- [ ] Rate limiting per endpoint
- [ ] API key rotation mechanism
- [ ] Input validatie op alle endpoints
- [ ] CORS policies voor web frontend
- [ ] SQL injection preventie (prepared statements)

## Deployment Pipeline

1. **Development**: Docker-compose lokaal
2. **Staging**: DigitalOcean droplet met staging database
3. **Production**: Dedicated server met:
   - MySQL master-slave replicatie
   - Redis cluster voor caching
   - Nginx reverse proxy
   - Automated backups (CoreAI3D native functionaliteit)

## Prioriteit

**Aanbevolen volgorde**: Fase 1 → Fase 2 → Fase 4 → Fase 5 → Fase 3

Reden: Zonder authenticatie is geen commercieel product mogelijk. Training pipeline is core differentiator. Container orchestratie maakt scaling mogelijk. Executable is distributiemodel. Website is presentation layer.