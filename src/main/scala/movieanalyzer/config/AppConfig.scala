package movieanalyzer.config

import com.typesafe.config.{Config, ConfigException, ConfigFactory}
import scala.collection.JavaConverters._
import scala.util.Try

/** Loads and provides typed access to application configuration.
  * Supports profile-based overrides (local, cluster) via HOCON.
  *
  * Purely functional — uses Option and Try instead of try/catch. */
case class AppConfig(
  sparkMaster: String,
  appName: String,
  dataDir: String,
  outputDir: String,
  // Hadoop config (only for cluster)
  hadoopFsDefault: Option[String],
  hadoopYarnRm: Option[String],
  hadoopYarnClasspath: Option[String],
  // Query params
  icebergMinMovies: Int,
  icebergMinAvgRating: Double,
  popularTagsMinRelevance: Double,
  popularTagsMinMovies: Int,
  popularTagsMinAvgRelevance: Double,
  skylineMinMovies: Int,
  reverseNnTargetMovieId: Int,
  reverseNnLikedThreshold: Double,
  anomalyTags: Seq[String],
  anomalyMinRelevance: Double,
  anomalyMaxRating: Double,
  topK: Int
)

object AppConfig {

  /** Loads configuration from HOCON files based on CLI args.
    *
    * @param args Command-line arguments (supports --profile local|cluster)
    * @return Fully populated AppConfig
    */
  def load(args: Array[String]): AppConfig = {
    val profile = parseProfile(args)
    val config = profile match {
      case Some(p) => ConfigFactory.load(s"$p.conf").withFallback(ConfigFactory.load())
      case None    => ConfigFactory.load()
    }
    fromConfig(config)
  }

  private def parseProfile(args: Array[String]): Option[String] = {
    val idx = args.indexWhere(a => a == "--profile" || a.startsWith("--profile="))
    if (idx < 0) None
    else if (args(idx).contains("=")) Some(args(idx).split("=", 2)(1))
    else if (idx + 1 < args.length) Some(args(idx + 1))
    else None
  }

  private def fromConfig(config: Config): AppConfig = {
    AppConfig(
      sparkMaster = config.getString("app.spark.master"),
      appName = config.getString("app.spark.app-name"),
      dataDir = config.getString("app.paths.data-dir"),
      outputDir = config.getString("app.paths.output-dir"),
      hadoopFsDefault = getOptionalString(config, "app.hadoop.fs.defaultFS"),
      hadoopYarnRm = getOptionalString(config, "app.hadoop.yarn.resourcemanager.address"),
      hadoopYarnClasspath = getOptionalString(config, "app.hadoop.yarn.application.classpath"),
      icebergMinMovies = config.getInt("app.queries.iceberg.min-movies"),
      icebergMinAvgRating = config.getDouble("app.queries.iceberg.min-avg-rating"),
      popularTagsMinRelevance = config.getDouble("app.queries.popular-tags.min-relevance"),
      popularTagsMinMovies = config.getInt("app.queries.popular-tags.min-movies"),
      popularTagsMinAvgRelevance = config.getDouble("app.queries.popular-tags.min-avg-relevance"),
      skylineMinMovies = config.getInt("app.queries.skyline.min-movies"),
      reverseNnTargetMovieId = config.getInt("app.queries.reverse-nn.target-movie-id"),
      reverseNnLikedThreshold = config.getDouble("app.queries.reverse-nn.liked-threshold"),
      anomalyTags = config.getStringList("app.queries.anomaly.tags").asScala.toList,
      anomalyMinRelevance = config.getDouble("app.queries.anomaly.min-relevance"),
      anomalyMaxRating = config.getDouble("app.queries.anomaly.max-rating"),
      topK = config.getInt("app.queries.top-k")
    )
  }

  /** Safely retrieves an optional config string using Try + toOption.
    * Pure — no exceptions escape. */
  private def getOptionalString(config: Config, path: String): Option[String] =
    Try(config.getString(path)).toOption
}
