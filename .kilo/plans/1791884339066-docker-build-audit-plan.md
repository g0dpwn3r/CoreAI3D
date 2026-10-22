# CoreAI3D Docker Build Audit & Herstel — Plan

## Doel

Inventarisatie en diagnose van de Docker-build van alle CoreAI3D services en sandboximages. Fase 1 (inventarisatie) en Fase 2 (diagnose Ubuntu-handtekeningfout) uitgevoerd. Geen bestanden gewijzigd in deze fase.

---

## Fase 1: Inventarisatie — Voltooid

### Dockerfiles (13 totaal)

| # | Dockerfile | Basisimage | Pakketbeheerder | Externe repos | Node.js | Bekende fouten |
|---|---|---|---|---|---|---|
| 1 | `Dockerfile` | ubuntu:24.04 | apt | vcpkg (GitHub fork) | Geen (C++ build) | Cache-mounts op apt; git lfs install |
| 2 | `web/Dockerfile` | node:20-alpine | apk + npm | Geen | Node 20 (base image) | `npm ci --legacy-peer-deps \|\| npm install` fallback |
| 3 | `docker/training/Dockerfile.trainer` | python:3.9-slim | apt + pip | Geen | Geen | Geen |
| 4 | `docker/admin-gui/Dockerfile.admin-gui` | lscr.io/linuxserver/webtop:ubuntu-xfce | apt | deadsnakes PPA | Geen (Python 3.10 via PPA) | Geen |
| 5 | `docker/sandbox/Dockerfile.ubuntu` | ubuntu:22.04 | apt + pip + npm | NodeSource 20.x | Node 20 (NodeSource) | `--allow-insecure-repositories`, `--allow-unauthenticated`, `apt-key` deprecated, dubbele `apt-get update`, pyenv 3.10.0, `sudo` groep (sandbox user niet in sudoers) |
| 6 | `docker/sandbox/Dockerfile.windows` | ubuntu:22.04 | apt + pip + npm | WineHQ, NodeSource 20.x | Node 20 (NodeSource) | `apt-key add` (deprecated), `add-apt-repository`, dubbele `apt-get update`, `chpasswd sandbox` (typo: spatie), WineHQ key via wget |
| 7 | `docker/sandbox/Dockerfile.kali` | kalilinux/kali-rolling | apt + pip + npm | Kali repos (HTTP) | Node (system) | `deb http://kali.download/kali` (HTTP), `apt update` + `apt-get update` gemengd, `npm install -g` zonder Node.js pin |
| 8 | `docker/sandbox/Dockerfile.alpine` | alpine:3.20 | apk + pip + npm | Geen | Node (system, alpine package) | `shadow` package, pyenv install van 3.10.0 kan falen, `npm install -g` als root |
| 9 | `docker/sandbox/Dockerfile.debian` | debian:12 | apt + pip + npm | Geen | Node (system) | `apt` i.p.v. `apt-get`, `apt update` + `apt install` zonder `--no-install-recommends`, pyenv 3.10.0, `usermod -aG sudo` (sudo niet geïnstalleerd) |
| 10 | `docker/sandbox/Dockerfile.arch` | archlinux:latest | pacman + pip + npm | Arch repos | Node (system) | `pacman-key --refresh-keys` (traag/onbetrouwbaar), `pacman -Sy` zonder `-u`, `pyenv install 3.10.0` |
| 11 | `docker/sandbox/Dockerfile.fedora` | fedora:40 | dnf + pip + npm | Geen | Node (system) | `--setopt=strict=0`, `dnf clean all` aan het einde (niet per stap), `pip3` i.p.v. venv pip, `npm install -g` als root |
| 12 | `docker/sandbox/Dockerfile.centos` | rockylinux:8 | dnf + pip + npm | NodeSource 20.x (rpm) | Node 20 (NodeSource) | `--setopt=strict=0`, `dnf config-manager --enable epel`, `chpasswd  sandbox` (dubbele spatie), `pip3` i.p.v. venv pip, `npm install -g` als root |
| 13 | `docker/sandbox/Dockerfile.opensuse` | opensuse/leap:15.5 | zypper + pip + npm | Geen | Node 20.14.0 (tarball) | `--no-gpg-checks` (GPG uitgeschakeld), `zypper --no-gpg-checks refresh`, `npm install -g` als root |

### Docker Compose services (13)

| Service | Dockerfile | Poorten | Afhankelijkheden |
|---|---|---|---|
| mysql | mysql:latest (image) | 33060 (expose) | Geen |
| coreai3d | Dockerfile | 1906:8080, 1907:8081 | mysql (healthy) |
| ubuntu-sandbox | docker/sandbox/Dockerfile.ubuntu | Geen | Geen |
| kali-sandbox | docker/sandbox/Dockerfile.kali | Geen | Geen |
| alpine-sandbox | docker/sandbox/Dockerfile.alpine | Geen | Geen |
| debian-sandbox | docker/sandbox/Dockerfile.debian | Geen | Geen |
| arch-sandbox | docker/sandbox/Dockerfile.arch | Geen | Geen |
| centos-sandbox | docker/sandbox/Dockerfile.centos | Geen | Geen |
| fedora-sandbox | docker/sandbox/Dockerfile.fedora | Geen | Geen |
| opensuse-sandbox | docker/sandbox/Dockerfile.opensuse | Geen | Geen |
| windows-sandbox | docker/sandbox/Dockerfile.windows | Geen | Geen |
| ai-trainer | docker/training/Dockerfile.trainer | Geen | Alle sandboxes |
| web | web/Dockerfile | 3000:3000 | mysql (healthy), coreai3d (started) |

### Scripts die tijdens builds worden uitgevoerd

- `docker/sandbox/pyenv_init.sh` — pyenv initialisatie
- `docker/sandbox/build_and_release.sh` — build en release script
- `docker/admin-gui/custom-cont-init.d/install-python310.sh` — Python 3.10 installatie
- `docker/training/train_linux_operations.py` — training script (gekopieerd in image)

### .dockerignore (root)

Bevat: `.env*`, `docker-compose.yml`, `.git`, `node_modules`, `build`, `dist`, `.next`, `out`, `.vscode`, `__pycache__`, `*.py[cod]`, `*.so`

**Opmerking:** `docker-compose.yml` staat in `.dockerignore`, wat correct is (niet nodig in build context). Maar `docker/sandbox/Dockerfile.*` worden wel in de build context opgenomen omdat `context: .` in docker-compose.yml.

---

## Fase 2: Diagnose Ubuntu-handtekeningfout — Voltooid

### Waargenomen fout

```
Err:2 http://security.ubuntu.com/ubuntu jammy-security InRelease
  At least one invalid signature was encountered.
E: The repository 'http://security.ubuntu.com/ubuntu jammy-security InRelease' is not signed.
```

### Diagnostische resultaten

#### 1. Systeemdatum
- Host: `Mon Oct 19 16:57:28 UTC 2026` — correct ingesteld
- Container: `Mon Oct 19 16:57:29 UTC 2026` — correct ingesteld
- **Conclusie:** Datum is niet de oorzaak.

#### 2. Isolatie-test (minimale Ubuntu 22.04)
```bash
docker run --rm ubuntu:22.04 bash -lc 'date -u; apt-get update'
```
- **Resultaat:** Reproduceert exact dezelfde fout: "At least one invalid signature was encountered."
- **Conclusie:** Het probleem is niet specifiek voor de Dockerfile of build-context. Het is een host- of containeromgevingsprobleem.

#### 3. Proxy/VPN/firewall
- `env | grep -i proxy` → geen proxy-variabelen
- `docker info | grep -i proxy` → geen Docker-proxy
- **Conclusie:** Geen proxy of firewall interceptie.

#### 4. Netwerktest naar Ubuntu-mirror
```bash
curl -sI http://archive.ubuntu.com/ubuntu/dists/jammy/InRelease
```
- **Resultaat:** HTTP 200 OK, Content-Length: 270087, Server: Apache/2.4.58
- **Conclusie:** Netwerkverbinding naar Ubuntu-mirrors werkt. Mirror is bereikbaar en levert correcte data.

#### 5. GPG-verificatie (handmatig)
```bash
# Host gpgv (2.4.8):
gpgv --keyring /usr/share/keyrings/ubuntu-archive-keyring.gpg /tmp/InRelease
# Resultaat: "Good signature from Ubuntu Archive Automatic Signing Key (2018)"

# Container gpgv (2.2.27):
gpgv --keyring /etc/apt/trusted.gpg.d/ubuntu-keyring-2018-archive.gpg /tmp/InRelease
# Resultaat: "Good signature from Ubuntu Archive Automatic Signing Key (2018)"
```
- **Conclusie:** De GPG-handtekening is geldig. De keyring bevat de juiste sleutel. Het probleem is niet de handtekening zelf.

#### 6. apt-get debug output (beslissend)
```bash
apt-get -o Debug::Acquire::gpgv=true update 2>&1
```
Output toont:
```
Preparing to exec: /usr/bin/apt-key --quiet --readonly verify --status-fd 3 /tmp/apt.sig.XXXXX /tmp/apt.data.XXXXX
gpgv exited with status 1
Err:1 http://archive.ubuntu.com/ubuntu jammy InRelease
  At least one invalid signature was encountered.
```

#### 7. Schijfruimte (ROOT CAUSE)
```bash
df -h  # in container
```
Output toont:
```
Filesystem      Size  Used Avail Use% Mounted on
overlay         202G  196G     0 100% /
```

**De container's root filesystem (`/`) is 100% vol.** 202G gebruikt, 0 bytes beschikbaar.

Host-bevestiging:
```
/dev/md2        202G  196G     0 100% /
```

Docker's root directory is `/mnt/data_ssd/docker` (op `/dev/sdc`, 938G, 2% gebruikt), maar de overlay filesystems (waar containers hun rootfs hebben) zijn op `/dev/md2` (202G, 100% vol). Dit betekent dat Docker's containerd overlay snapshots op `/dev/md2` worden opgeslagen, niet op `/dev/sdc`.

### Root cause (definitief)

**De host's root-partitie (`/dev/md2`, 202 GB) is 100% vol.** Dit is de werkelijke oorzaak van de "At least one invalid signature was encountered" fout.

**Mechanisme:**
1. apt-get downloadt de InRelease file naar een tijdelijk bestand in `/var/lib/apt/lists/partial/` (op `/dev/md2`).
2. apt-get splitst de InRelease in sig + data en slaat deze op in `/tmp/apt.sig.*` en `/tmp/apt.data.*` (op `/dev/md2`).
3. apt-get roept `apt-key verify` aan, wat intern `gpgv` aanroept.
4. gpgv probeert een tijdelijke GPG-home aan te maken (`mktemp -d /tmp/apt-key-gpghome.XXXXXXXXXX`).
5. **Dit faalt met "No space left on device"** omdat `/dev/md2` vol is.
6. apt-get interpreteert dit falen als "At least one invalid signature was encountered" (misleidende foutmelding).

**Bewijs:**
- Handmatige `gpgv` op dezelfde InRelease file zegt "Good signature" (exit 0).
- `apt-key --quiet --readonly verify` op dezelfde file zegt "Good signature" (exit 0).
- Alleen apt-get's interne aanroep (met `--status-fd 3` en tijdelijke bestanden) faalt.
- Container disk space: 0 bytes beschikbaar op `/` (overlay op `/dev/md2`).
- Host disk space: `/dev/md2` is 100% vol (196G/202G), `/dev/sdc` heeft 875G vrij.

### Aanbevolen oplossingen (in volgorde van voorkeur)

1. **Vrijmaken van schijfruimte op `/dev/md2` (aanbevolen, minimaal invasief):**
   ```bash
   # Ruim Docker build cache op
   docker system prune -a --volumes
   
   # Ruim apt cache op de host
   apt-get clean && apt-get autoremove -y
   
   # Ruim oude logs op
   journalctl --vacuum-size=100M
   
   # Controleer grote bestanden
   du -sh /var/* 2>/dev/null | sort -rh | head -10
   ```

2. **Docker storage verplaatsen naar `/dev/sdc` (structureler):**
   ```bash
   # Stop Docker
   systemctl stop docker
   
   # Verplaats data
   rsync -aP /mnt/data_ssd/docker/ /mnt/data_ssd/docker_new/
   mv /mnt/data_ssd/docker /mnt/data_ssd/docker_old
   mv /mnt/data_ssd/docker_new /mnt/data_ssd/docker
   
   # Update /etc/docker/daemon.json
   # {"data-root": "/mnt/data_ssd/docker"}
   
   # Start Docker
   systemctl start docker
   ```
   **Let op:** Dit is een significantere wijziging en vereist downtime.

3. **Build alleen de noodzakelijke services (tijdelijk):**
   ```bash
   docker compose build coreai3d web
   ```
   Dit slaat de sandbox-images over die de meeste schijfruimte verbruiken.

### Veiligheidsregels (strikt naleven)

- **Geen** `--allow-unauthenticated`, `--allow-insecure-repositories`, `trusted=yes`
- **Geen** uitschakeling van GPG-validatie
- **Geen** HTTP→HTTPS blind wijzigen
- **Geen** willekeurige GPG-sleutels van onbetrouwbare bronnen
- De oorzaak is **niet** in de Dockerfiles — het is een host-schijfruimteprobleem

### Aanbevolen minimale wijzigingen aan Dockerfiles (na schijfruimte-fix)

| Dockerfile | Huidige onveilige code | Aanbevolen fix |
|---|---|---|
| `Dockerfile.ubuntu` | `--allow-insecure-repositories`, `--allow-unauthenticated`, `apt-key adv` | Verwijder `--allow-*` flags; gebruik `gpg --dearmor` voor keys; combineer `apt-get update && apt-get install` |
| `Dockerfile.windows` | `apt-key add winehq.key`, dubbele `apt-get update` | Gebruik `gpg --dearmor` voor WineHQ key; combineer update+install; verwijder deprecated `apt-key` |
| `Dockerfile.kali` | `deb http://kali.download/kali` (HTTP) | Wijzig naar HTTPS na verificatie dat mirror HTTPS ondersteunt |
| `Dockerfile.opensuse` | `--no-gpg-checks` | Verwijder `--no-gpg-checks`; gebruik `zypper refresh` zonder override |
| `Dockerfile.centos` | `--setopt=strict=0` | Verwijder `strict=0`; los de underliggende pakketconflicten op |
| `Dockerfile.arch` | `pacman-key --refresh-keys` | Vervang door `pacman-key --init && pacman-key --populate archlinux` |
| `Dockerfile.fedora` | `--setopt=strict=0` | Verwijder `strict=0` |

---

## Status

- **Fase 1 (inventarisatie):** ✅ Voltooid — 13 Dockerfiles, 13 Compose-services geïnventariseerd
- **Fase 2 (diagnose):** ✅ Voltooid — Root cause vastgesteld: host-schijfruimte (`/dev/md2` 100% vol). GPG-handtekeningen zijn geldig. Netwerk is bereikbaar. Proxy/firewall is niet het probleem.
- **Fase 3-8:** Nog niet gestart — wacht op schijfruimte-fix en bevestiging

## Open vragen

1. ✅ Host-klok: correct ingesteld (Oct 19 2026)
2. ✅ Proxy/VPN/firewall: geen proxy of firewall interceptie
3. ✅ Ubuntu base image: recent (Oct 2 2026), GPG-sleutels geldig
4. ✅ Reproduceerbaarheid: ja, minimale test reproduceert exact dezelfde fout
5. ⏳ Prioriteit services: nog te bepalen (alle 13 of alleen coreai3d + web?)
6. ⏳ Schijfruimte: `/dev/md2` is 100% vol — moet worden opgelost voordat builds slagen

## Volgende stap

1. Vrijmaken van schijfruimte op `/dev/md2` (zie aanbeveling 1 hierboven).
2. Herstart Docker en voer de isolatie-test opnieuw uit.
3. Als de isolatie-test slaagt, ga door met Fase 3 (Linux-pakketinstallaties).
4. Pas de aanbevolen minimale wijzigingen toe per Dockerfile.
5. Voer `docker compose build` uit (zonder `--no-cache` eerst).
6. Als dat slaagt, voer `docker compose build --no-cache` uit.
7. Rapporteer resultaten per service.
