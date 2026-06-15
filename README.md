# Duluin Machine Learning

## Installation Tutorial

### Local Environment Setup

Firstly, make sure _pipenv_ library already installed on your machine. If not, run this command below:

```
$ pip install pipenv
```

Afterwards, don't forget to activate local environment with this command:

```
$ pipenv shell
```

If this is your first time, run the following command to install all libraries and dependencies:

```
$ pipenv install
```

(Optional) later, if any library is needed to install. Run similar command with additional library name, for instance:

```
$ pipenv install <library-name>
```

### Populate `.env` file

Make a copy of .env.example and then populate it according to your configurations

```
$ cp .env.example .env
```

### Run migration file to sync your database

Run this command below to sync your current database schemas to the updated schemas

```
$ alembic upgrade head
```

(Optional) if you need update your current migration. Firstly, add your model import statement in `app/models/__init__.py`. Afterwards, run this following command to autogenerate migration file

```
$ alembic revision --autogenerate -m "description"
```

### Seed placeholders data

In cases where there are data placeholders that need to be populated in the database, create and format them as JSON schemas in the `seeds/data` directory. Afterwards, create seeder class and make several changes in `seeds/seed.py` (see further instructions on that file). Lastly, run this command below

```
$ python seeds/seed.py
```

### Run Program

Execute this following code to run program with default port in 8000

```
$ uvicorn main:app --reload
```

## Logging

Application logs are emitted as JSON to stdout and file, so they are visible in `docker logs`.

Optional environment variables:

```
LOG_MASKING_ENABLED=true
LOG_MASK_FIELDS=nik,token,password,secret
LOG_UNMASK_FIELDS=company_id
LOG_MAX_VALUE_LENGTH=2000
LOG_MAX_COLLECTION_ITEMS=30
LOG_MAX_BODY_PREVIEW_LENGTH=1000
LOG_PRETTY_JSON=false
```

`LOG_MASK_FIELDS` adds more masked keys. `LOG_UNMASK_FIELDS` removes keys from the masking list.
`LOG_MAX_VALUE_LENGTH` limits long string values in logs. `LOG_MAX_COLLECTION_ITEMS` limits logged items in arrays/maps. `LOG_MAX_BODY_PREVIEW_LENGTH` limits raw body preview length.
`LOG_PRETTY_JSON=true` makes stdout/file logs multi-line formatted JSON for easier manual reading.
