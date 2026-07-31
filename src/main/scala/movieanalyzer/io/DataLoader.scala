package movieanalyzer.io

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.col

/** Container for all loaded datasets, providing both RDD and DataFrame access. */
case class DataBundle(
  // DataFrames (for Q6-Q10) — column names match original: MovieId, UserId, Rating, TagId, Relevance, Tag
  moviesDF: DataFrame,
  ratingsDF: DataFrame,
  tagsDF: DataFrame,
  genomeScoresDF: DataFrame,
  genomeTagsDF: DataFrame,
  // RDDs (for Q1-Q5)
  moviesRDD: RDD[(String, String, Array[String])],     // (movieId, title, genres)
  ratingsRDD: RDD[(String, String, Double)],            // (userId, movieId, rating)
  tagsRDD: RDD[(String, String, String)],               // (userId, movieId, tag)
  genomeScoresRDD: RDD[(String, String, Double)],       // (movieId, tagId, relevance)
  genomeTagsRDD: RDD[(String, String)]                  // (tagId, tagName)
)

/** Unified data loading from CSV files with proper schemas and caching. */
object DataLoader {
  // Schemas with uppercase column names to match original query code
  private val ratingsSchema = StructType(Seq(
    StructField("UserId", StringType, nullable = false),
    StructField("MovieId", StringType, nullable = false),
    StructField("Rating", DoubleType, nullable = false),
    StructField("Timestamp", StringType, nullable = true)
  ))

  private val tagsSchema = StructType(Seq(
    StructField("UserId", StringType, nullable = false),
    StructField("MovieId", StringType, nullable = false),
    StructField("Tag", StringType, nullable = true),
    StructField("Timestamp", StringType, nullable = true)
  ))

  private val genomeScoresSchema = StructType(Seq(
    StructField("MovieId", StringType, nullable = false),
    StructField("TagId", StringType, nullable = false),
    StructField("Relevance", DoubleType, nullable = false)
  ))

  /** Loads all MovieLens datasets from the specified directory.
    *
    * @param spark   Active SparkSession
    * @param dataDir Base directory containing CSV files (local FS or HDFS)
    * @return DataBundle with cached DataFrames and derived RDDs
    */
  def load(spark: SparkSession, dataDir: String): DataBundle = {
    println(s"Loading datasets from: $dataDir")

    // Load DataFrames with proper CSV options for quoted fields
    val moviesDF = spark.read
      .option("header", "true")
      .option("quote", "\"")
      .option("escape", "\"")
      .csv(s"$dataDir/movies.csv")
      .select(
        col("movieId").alias("MovieId"),
        col("title").alias("Title"),
        col("genres").alias("Genres")
      )
      .cache()

    val ratingsDF = spark.read
      .option("header", "true")
      .schema(ratingsSchema)
      .csv(s"$dataDir/ratings.csv")
      .select("UserId", "MovieId", "Rating")
      .cache()

    val tagsDF = spark.read
      .option("header", "true")
      .option("quote", "\"")
      .option("escape", "\"")
      .schema(tagsSchema)
      .csv(s"$dataDir/tags.csv")
      .select("UserId", "MovieId", "Tag")
      .cache()

    val genomeScoresDF = spark.read
      .option("header", "true")
      .schema(genomeScoresSchema)
      .csv(s"$dataDir/genome-scores.csv")
      .cache()

    val genomeTagsDF = spark.read
      .option("header", "true")
      .option("quote", "\"")
      .option("escape", "\"")
      .csv(s"$dataDir/genome-tags.csv")
      .select(
        col("tagId").alias("TagId"),
        col("tag").alias("Tag")
      )
      .cache()

    // Derive RDDs from DataFrames (consistent parsing, no manual string splitting)
    val moviesRDD = moviesDF.rdd.map { row =>
      val movieId = row.getAs[String]("MovieId")
      val title = row.getAs[String]("Title")
      val genresStr = row.getAs[String]("Genres")
      val genres = if (genresStr == null || genresStr == "(no genres listed)")
        Array.empty[String]
      else
        genresStr.split("\\|")
      (movieId, title, genres)
    }.filter(_._3.nonEmpty)

    val ratingsRDD = ratingsDF.rdd.map { row =>
      (row.getAs[String]("UserId"), row.getAs[String]("MovieId"), row.getAs[Double]("Rating"))
    }

    val tagsRDD = tagsDF.rdd.map { row =>
      (row.getAs[String]("UserId"), row.getAs[String]("MovieId"), row.getAs[String]("Tag"))
    }

    val genomeScoresRDD = genomeScoresDF.rdd.map { row =>
      (row.getAs[String]("MovieId"), row.getAs[String]("TagId"), row.getAs[Double]("Relevance"))
    }

    val genomeTagsRDD = genomeTagsDF.rdd.map { row =>
      (row.getAs[String]("TagId"), row.getAs[String]("Tag"))
    }

    println("All datasets loaded and cached.")
    DataBundle(
      moviesDF, ratingsDF, tagsDF, genomeScoresDF, genomeTagsDF,
      moviesRDD, ratingsRDD, tagsRDD, genomeScoresRDD, genomeTagsRDD
    )
  }
}
