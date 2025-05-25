#Movie Preference Analyzer

A Scala + Apache Spark project for **functional big data analytics** on the [MovieLens](https://grouplens.org/datasets/movielens/) dataset, developed as part of the INF424 course: *Functional Programming, Analytics and Applications* at the **Technical University of Crete** (Spring Semester 2024–2025).

This project implements a variety of advanced queries using both **Spark RDDs** and **DataFrames**, with all code written in Scala and tested on the SoftNet Cluster of TUC.

Authors:
- @Asterinos1
- @eNiaro

## Deployment Modes

The tool supports two execution environments:

- **Cluster Mode** – Optimized for execution on the **SoftNet Cluster**.
- **Local Mode** – Designed for development and testing on a local machine, using local file paths and standalone Spark execution. Requires a local installation of **Apache Spark**, **Hadoop**, and **Java JDK** (to support Scala execution).

## Dataset

Dataset used: [ml-latest.zip](https://files.grouplens.org/datasets/movielens/ml-latest.zip)

CSV files used:
- `movies.csv` – movie metadata (movieId, title, genres)
- `ratings.csv` – user ratings (userId, movieId, rating)
- `tags.csv` – user-assigned tags
- `genome-scores.csv` – tag relevance scores per movie
- `genome-tags.csv` – textual tag labels



## Technologies
- Scala
- Apache Spark
- HDFS (SoftNet Cluster)

## Build Configuration

This project uses **SBT** for managing dependencies and building the application. Below is the configuration from `build.sbt`:

```scala
ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.11.8"

lazy val root = (project in file("."))
  .settings(
    name := "ScalaAnalyticsProject"
  )

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.3.1",
  "org.apache.spark" %% "spark-sql" % "2.3.1",
  "org.apache.hadoop" % "hadoop-client" % "3.1.1"
)
```

**Check out `project_doc.pdf` for the full documentation of this project.**
