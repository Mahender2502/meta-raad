# Chroma server with the prebuilt N24 News index baked in (read-only retrieval data).
#
# Build from the repo root, after unzipping chroma_n24news.zip into
# data/embeddings/chroma/ (chroma.sqlite3 must sit directly in that folder):
#   docker build -f docker/chroma-n24/Dockerfile -t ghcr.io/<user>/meta-raad-chroma:n24-v1 .
#   docker push ghcr.io/<user>/meta-raad-chroma:n24-v1
#
# The base image persists to /data (see its /config.yaml) and declares no VOLUME,
# so data copied there is kept. Writes made at runtime are lost when the
# container is removed; re-index and push a new tag to change the data.
FROM chromadb/chroma:1.4.4
COPY data/embeddings/chroma /data
