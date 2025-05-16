import org.apache.spark.sql.SparkSession

object MoviePreferenceAnalyzer extends App {

  val spark = SparkSession.builder()
    .appName("MoviePreferenceAnalyzer")
    .master("local[*]")
    .getOrCreate()

  val sc = spark.sparkContext

  //"hdfs://localhost:9000/ml-latest/"
  val ratingsPath = "hdfs://localhost:9000/ml-latest/ratings.csv"
  val tagsPath = "hdfs://localhost:9000/ml-latest/tags.csv"
  val genomeScoresPath = "hdfs://localhost:9000/ml-latest/genome-scores.csv"
  val linksPath = "hdfs://localhost:9000/ml-latest/links.csv"
  val moviesPath = "hdfs://localhost:9000/ml-latest/movies.csv"
  val genomeTagsPath = "hdfs://localhost:9000/ml-latest/genome-tags.csv"

  //Creating rdds here
  val movies = sc.textFile(moviesPath)
    .filter(line => !line.startsWith("movieId"))
    .map((line: String) => line.split(",", 3))  // title might contain commas
    .map(fields => (fields(0), fields(1), fields(2).split('|').toList))  // (movieId, List(genres))

  val ratings = sc.textFile(ratingsPath)
    .filter(line => !line.startsWith("userId"))  //skip the csv headers.
    .map((line:String)=>line.split(","))            //break the string into its components
    .map(fields => ((fields(0), fields(1)), fields(2).toDouble))  // ((userId, movieId), rating)

  val tags = sc.textFile(tagsPath)
    .filter(line => !line.startsWith("userId"))
    .map((line:String)=>line.split(","))
    .map(fields => ((fields(0), fields(1)), fields(2)))  // ((userId, movieId), tag)

  //Query 1: Iceberg Query — Top Tags by Genre with High Ratings
  //Objective: Find genre-tag pairs that appear in more than 100 movies and have an average user
  //rating greater than 4.0.

  val avgRatingsAboveFour = ratings
    .map { case ((userId, movieId), rating) => (movieId, (rating, 1)) }  // movieId → (rating, 1)
    .reduceByKey((a: (Double, Int), b: (Double, Int)) => (a._1 + b._1, a._2 + b._2))
    .mapValues { case (sum, count) => sum / count }
    .filter(_._2 > 4)

  // getting movieId - tag pairs (dropping the userId)
  val tagsByMovie = tags.map { case ((userId, movieId), tag) => (movieId, tag) }
  val aboveFourTaggedMovies = avgRatingsAboveFour.join(tagsByMovie)

  val movieGenres = movies.map { case (mid, title, genres) => (mid, genres) }
  val joinedWithGenres = aboveFourTaggedMovies.join(movieGenres)
  // => (movieId, ((avgRating, tag), List(genre)))

  val genreTagRatings = joinedWithGenres.flatMap{
    case (_, ((avgRating, tag), genres)) =>
      genres.map(genre => ((genre, tag), (avgRating, 1)))  // flattening the genres.
  }.reduceByKey((a: (Double, Int), b: (Double, Int)) => (a._1 + b._1, a._2 + b._2))

  val icebergResults = genreTagRatings
    .filter { case (_, (_, count)) => count > 100 }
    .mapValues { case (total, count) => (total / count, count) }

  icebergResults.saveAsTextFile("hdfs://localhost:9000/ml-latest/query1_output")

  // End of Query 1
  // ==============================================================


  //Query 4: Sentiment Estimation — Tags Associated with Ratings
  //Objective: Compute the average user rating for each user-assigned tag from tags.csv

  val averageOfEachTag =
      tags.join(ratings)
      .map {case (_, (tag, rating)) => (tag, (rating, 1))}
      .reduceByKey((a: (Double, Int), b: (Double, Int)) => (a._1 + b._1, a._2 + b._2))
      .mapValues { case (total, count) => total / count}

  averageOfEachTag.saveAsTextFile("hdfs://localhost:9000/ml-latest/query4_output")
  // debugging purposes.
  // averageOfEachTag.foreach { case (tag, avg) => println(f"$tag%-30s -> $avg%.2f") }

  // End of Query 4
  // ==============================================================

  spark.stop()
}
