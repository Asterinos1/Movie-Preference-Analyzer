ThisBuild / version      := "1.0.0"
ThisBuild / scalaVersion := "2.12.18"
ThisBuild / organization := "gr.tuc.softnet"

lazy val root = (project in file("."))
  .settings(
    name := "movie-preference-analyzer",

    libraryDependencies ++= Seq(
      // Spark (provided scope — cluster already has these)
      "org.apache.spark" %% "spark-core" % "3.5.4" % "provided",
      "org.apache.spark" %% "spark-sql"  % "3.5.4" % "provided",

      // Configuration management
      "com.typesafe" % "config" % "1.4.3",

      // Testing
      "org.scalatest"    %% "scalatest"  % "3.2.18" % Test,
      "org.apache.spark" %% "spark-core" % "3.5.4"  % Test,
      "org.apache.spark" %% "spark-sql"  % "3.5.4"  % Test
    ),

    // Assembly settings for fat JAR (cluster deployment)
    assembly / mainClass := Some("movieanalyzer.Main"),
    assembly / assemblyMergeStrategy := {
      case PathList("META-INF", _*) => MergeStrategy.discard
      case _                        => MergeStrategy.first
    },

    // Compiler options
    scalacOptions ++= Seq(
      "-deprecation",
      "-feature",
      "-unchecked"
    ),

    // Fork JVM for tests (required for Spark)
    Test / fork := true,
    Test / javaOptions ++= Seq(
      "-Xmx2g",
      "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
      "--add-opens=java.base/java.lang=ALL-UNNAMED",
      "--add-opens=java.base/java.util=ALL-UNNAMED"
    )
  )
