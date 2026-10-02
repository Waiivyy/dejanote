# Docker eating disk space

The laptop ran out of space again. Docker was using 60 GB.

See what is taking up room:

```bash
docker system df
```

Remove stopped containers, unused networks, dangling images and build cache:

```bash
docker system prune
```

Add `-a` to also remove images not used by any container (they get downloaded again when needed), and `--volumes` to drop unused volumes. Careful with volumes: the local Postgres data lives in one.

On macOS the disk image does not shrink by itself after pruning. In the Docker Desktop settings, lower the disk limit under Resources, or reset it if nothing local matters.
