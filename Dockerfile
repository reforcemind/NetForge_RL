FROM python:3.12-slim

WORKDIR /netforge

COPY . /netforge

RUN pip install --no-cache-dir -e .

CMD ["python", "-c", "from netforge_rl.environment.parallel_env import NetForgeRLEnv"]
