# ============================================================
# Multi-stage Dockerfile for Movie Preference Analyzer
# Stage 1: Build fat JAR via SBT
# Stage 2: Spark runtime with spark-submit
# ============================================================

# --- Stage 1: Build ---
FROM sbtscala/scala-sbt:eclipse-temurin-jammy-11.0.22_7_1.9.9_2.12.18 AS builder

WORKDIR /build

# Cache dependency resolution (these layers change rarely)
COPY build.sbt .
COPY project/build.properties project/
COPY project/plugins.sbt project/

RUN sbt update

# Copy source and compile fat JAR
COPY src/ src/

RUN sbt assembly

# --- Stage 2: Runtime ---
FROM spark:3.5.4

USER root

# Create app and data directories
RUN mkdir -p /app /data/ml-latest /output

# Copy fat JAR from builder
COPY --from=builder /build/target/scala-2.12/movie-preference-analyzer-assembly-*.jar /app/movie-preference-analyzer.jar

# Copy config files (for reference / override)
COPY src/main/resources/ /app/conf/

# Set ownership (official Spark image uses uid 185)
RUN chown -R 185:185 /app /data /output

USER 185

ENV SPARK_HOME=/opt/spark

ENTRYPOINT ["/opt/spark/bin/spark-submit", \
  "--class", "movieanalyzer.Main", \
  "--master", "local[*]", \
  "--driver-memory", "4g", \
  "/app/movie-preference-analyzer.jar"]

# Default args: run all queries with docker profile
CMD ["--profile", "docker", "--query", "all"]
