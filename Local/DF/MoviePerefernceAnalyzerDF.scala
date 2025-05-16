import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object MoviePreferenceAnalyzerDF extends App {

  val spark = SparkSession.builder()
    .appName("MoviePreferenceAnalyzerDF")
    .master("local[*]")
    .getOrCreate()

  val ratingsPath = "hdfs://localhost:9000/ml-latest/ratings.csv"
  val tagsPath = "hdfs://localhost:9000/ml-latest/tags.csv"
  val genomeScores = "hdfs://localhost:9000/ml-latest/genome-scores.csv"
  val links = "hdfs://localhost:9000/ml-latest/links.csv"
  val moves = "hdfs://localhost:9000/ml-latest/movies.csv"
  val genomeTags = "hdfs://localhost:9000/ml-latest/genome-tags.csv"

  val query6_output = "hdfs://localhost:9000/ml-latest/query6_output"
  val query9_output = "hdfs://localhost:9000/ml-latest/query9_output"

  val df_gt = spark.read
    .option("header", "true")
    .csv(genomeTags)

  val df_ratings = spark.read
    .option("header", "true")
    .csv(ratingsPath)

  val df_gs = spark.read
    .option("header", "true")
    .csv(genomeScores)

  //Query 6: Skyline Query — Non-Dominated Movies in Multiple Dimensions
  //Objective: Identify movies that are not dominated in average rating, rating count, and average tag relevance.
  val movieRatings = df_ratings
    .select(col("movieId").cast("int"), col("rating").cast("double"))
    .groupBy("movieId")
    .agg(
      avg("rating").as("avg_rating"),
      count("rating").as("rating_count")
    )

  val movieRelevance = df_gs
    .select(col("movieId").cast("int"), col("relevance").cast("double"))
    .groupBy("movieId")
    .agg(avg("relevance").as("avg_relevance"))

  //combine the previous dataframes to get our complete dataframe
  val movieStats = movieRatings.join(movieRelevance, "movieId")

  //we will utilize the self join operation
  //create 2 copies of the same dataframe.
  //we will name them a and b, and compare each line
  val statsA = movieStats.alias("a")
  val statsB = movieStats.alias("b")

  val dominationCondition =
    (col("b.avg_rating") >= col("a.avg_rating")) &&   //check if a row in "b" is better than a row in "a"
      (col("b.rating_count") >= col("a.rating_count")) &&
      (col("b.avg_relevance") >= col("a.avg_relevance")) &&
      (       //and here we check if it is strictly better in one of the attributes
        (col("b.avg_rating") > col("a.avg_rating")) ||
          (col("b.rating_count") > col("a.rating_count")) ||
          (col("b.avg_relevance") > col("a.avg_relevance"))
        )
  //here we will get a coloumn in retourn with true/false statements regarding the comparissons.
  //we will use this column to perform a left_anti join to exclude the dominated movies
  //(we are not interested in performing a plain left since we will get nulls in return)
  val skylineDF = statsA.join(statsB, dominationCondition, "left_anti")

  //skylineDF.show()
  skylineDF.rdd.saveAsTextFile(query6_output)

  //Query 9: Iceberg Query — Top Tags by Genre with High Ratings
  //Objective: Find genre-tag pairs that appear in more than 100 movies and have an average user
  //rating greater than 4.0.

  val famous_tags = df_gt //we filter only the tagid for the desired tags
    .filter(col("tag").isin("action", "classic", "thriller"))

  val avgRatings = df_ratings
    .select(col("movieId").cast("int"), col("rating").cast("double")) //cast types
    .groupBy("movieId")                                               //group by each movie
    .agg(avg("rating").as("avg_rating"))     //for each movie entry, calculate the avg on column ratings

  val overhypedMovies = famous_tags.join(df_gs, "tagId")  //we join the genome scores table to the filtered tags
    .filter(col("relevance").>=(0.8))   //we keep only the desired relevance
    .groupBy("movieId")                 //then calculate the average relevance of each movie
    .agg(avg("relevance").as("avg_relevance_to_tags"))
    .join(avgRatings, "movieId")    //join the average ratings table
    .filter(col("avg_rating") < 2.5)  //filter those with average rating lower than 2.5

//  overhypedMovies.show()
  overhypedMovies.rdd.saveAsTextFile(query9_output)

  spark.stop()
}
