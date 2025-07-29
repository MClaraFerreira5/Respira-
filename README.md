# Respira+ — Backend API (FastAPI + PostgreSQL)

Este projeto é o backend da aplicação **Respira+**, desenvolvido com FastAPI, SQLAlchemy, PostgreSQL e Alembic para controle de versão do banco de dados.

## Criando o database

Inicialmente crie, através do pgAdmin, o database `respira_mais`

## Executando a aplicação

Para rodar a API localmente:

```bash
uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload
````

## Gerenciando as Migrations

Usamos o Alembic para versionamento de schema do banco de dados. As migrations são criadas automaticamente a partir dos modelos definidos.

### Adicionando um novo modelo

1. Importe o modelo no arquivo `app/db/base_class_imports.py`, por exemplo:

```python
from models.user_model import User
````

2. Gere a migration com:

```bash
alembic revision --autogenerate -m "Adicionando modelo de usuário"
````

3. Aplique a migration ao banco:

```bash
alembic upgrade head
````

## Testando a API

Acesse a documentação interativa do FastAPI em:

```bash
http://localhost:8000/docs
````