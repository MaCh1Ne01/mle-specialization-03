# Estrategia de Git utilizada

## Branching Model
Se usó una adaptación simplificada de [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow) con las siguientes ramas:
- main: Versión estable (Producción).  
- development: Versión con cambios por testear (Desarrollo).

## Convenciones
**Pull Requests**:  
- Requieren revisión de al menos 1 colaborador.  

## Comandos
```bash
# Clonar un repositorio remoto en local:
git clone repository-url

# Crear y cambiarse a nueva branch:
git checkout -b branch-name

# Ver branch en la que se encuentra:
git branch

# Muestra el estado actual del repositorio (rama actual, cambios no reastreados, conflictos, etc.):
git status

# Establece el nombre y email del autor de los commits:
git config --global user.name "your_name"
git config --global user.email "your_email"

# Cambiar de branch:
git checkout branch-name

# Prepara todos los cambios para el próximo commit:
git add .

# Guarda los cambios preparados para ser subidos al repositorio remoto:
git commit -m "commit-message"

# Sube los cambios commiteados al repositorio remoto:
git push origin branch-name
```