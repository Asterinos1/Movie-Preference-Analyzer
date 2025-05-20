import org.apache.spark.sql.SparkSession
import org.apache.hadoop.fs.FileSystem
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{avg, col, count, row_number, sqrt, sum}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

object Main extends App{
  val spark = SparkSession.builder
    .appName("ScalaProject")
    .master("yarn")
    .config("spark.hadoop.fs.defaultFS", "hdfs://clu01.softnet.tuc.gr:8020")
    .config("spark.hadoop.yarn.resourcemanager.address", "http://clu01.softnet.tuc.gr:8189")
    .config("spark.hadoop.yarn.application.classpath",
      "$HADOOP_CONF_DIR,$HADOOP_COMMON_HOME/*," +
        "$HADOOP_COMMON_HOME/lib/*,$HADOOP_HDFS_HOME/*," +
        "$HADOOP_HDFS_HOME/lib/*,$HADOOP_MAPRED_HOME/*," +
        "$HADOOP_MAPRED_HOME/lib/*,$HADOOP_YARN_HOME/*," +
        "$HADOOP_YARN_HOME/lib/*")
    .getOrCreate()

  val hdfsURI = "hdfs://clu01.softnet.tuc.gr:8020"
  FileSystem.setDefaultUri(spark.sparkContext.hadoopConfiguration, hdfsURI)
  val hdfs = FileSystem.get(spark.sparkContext.hadoopConfiguration)

  val dataPath = "/user/chrisa/ml-latest/"
  val outputPath = "/user/fp25_1"

  //Defining the Paths of the data
  val moviesPathHDFS = dataPath + "movies.csv"
  val ratingsPathHDFS = dataPath + "ratings.csv"
  val tagsPathHDFS = dataPath + "tags.csv"
  //val linksPathHDFS = dataPath + "links.csv"        NOT NEEDED
  val genomeTagsPathHDFS = dataPath + "genome-tags.csv"
  val genomeScoresPathHDFS = dataPath + "genome-scores.csv"

  val outputPathQuery6 = outputPath + "/ProjectOutputs/Query6"
  val outputPathQuery7 = outputPath + "/ProjectOutputs/Query7"
  val outputPathQuery8 = outputPath + "/ProjectOutputs/Query8"
  val outputPathQuery9 = outputPath + "/ProjectOutputs/Query9"
  val outputPathQuery10 = outputPath + "/ProjectOutputs/Query10"

  val sc = spark.sparkContext

  //Defining Schemas
  val ratingsSchema = StructType(Seq(
    StructField("UserId", StringType, nullable = false),
    StructField("MovieId", StringType, nullable = false),
    StructField("Rating", DoubleType, nullable = false)
  ))

  val tagSchema = StructType(Seq(
    StructField("UserId", StringType, nullable = false),
    StructField("MovieId", StringType, nullable = false),
    StructField("Tag", StringType, nullable = false)
  ))

  val genomeScoresSchema = StructType(Seq(
    StructField("MovieId", StringType, nullable = false),
    StructField("TagId", StringType, nullable = false),
    StructField("Relevance", DoubleType, nullable = false)
  ))

  //Defining Dataframes
  val moviesFileDF = spark.read
    .option("header", "true")
    .option("quote", "\"")
    .option("escape", "\"")
    .csv(moviesPathHDFS)

  val ratingsFileDF = spark.read
    .option("header", "true")
    .schema(ratingsSchema)
    .csv(ratingsPathHDFS)

  val tagsFileDF = spark.read
    .option("header", "true")
    .option("quote", "\"")
    .schema(tagSchema)
    .csv(tagsPathHDFS)

  val genomeScoresDF = spark.read
    .option("header", "true")
    .schema(genomeScoresSchema)
    .csv(genomeScoresPathHDFS)

  val genomeTagsDF = spark.read
    .option("header","true")
    .csv(genomeTagsPathHDFS)

  /****************Creating RDDs here************************/

  //movies.csv
  //We need to make the rdd this way for this file because some movies have commas
  val moviesRDD = moviesFileDF.select(
      col("movieId").alias("MovieId"),
      col("title").alias("Title"),
      col("genres").alias("Genre")
    ).rdd
    .map(row => (row.getAs[String]("MovieId"), row.getAs[String]("Title"), row.getAs[String]("Genre").split("\\|")))

  //ratings.csv
  val ratingsRDD = ratingsFileDF
    .rdd
    .map(row => (row.getString(0), row.getString(1), row.getDouble(2)))


  //tags.csv
  //We need to make the rdd this way for this file because some tags have commas
  val tagsRDD = tagsFileDF
    .select("UserId", "MovieId", "Tag")
    .rdd
    .map(row => (row.getAs[String]("UserId"), row.getAs[String]("MovieId"), row.getAs[String]("Tag"))) //(userId, movieId, tag)

  //genome-scores.csv
  val rawGenomeScoresRDD = sc.textFile(genomeScoresPathHDFS)
  val genomeScoresHeader = rawGenomeScoresRDD.first()

  val genomeScoresRDD = genomeScoresDF
    .rdd
    .map(row => (row.getString(0), row.getString(1), row.getDouble(2)))

  //genome-tags.csv
  val rawGenomeTagsRDD = sc.textFile(genomeTagsPathHDFS)
  val genomeTagsHeader = rawGenomeTagsRDD.first()

  val genomeTagsRDD = rawGenomeTagsRDD
    .filter(row => row != genomeTagsHeader)          //skip the csv headers.
    .map(line => line.split(",",2))                  //break the string into its components
    .map(fields => (fields(0), fields(1)))           //(tagId,tag)

  //Query 6: Skyline Query — Non-Dominated Movies in Multiple Dimensions
  //Files Used: ratings.csv, genome-scores.csv
  //Objective: Identify movies that are not dominated in average rating, rating count,
  //and average tag relevance.
  val movieRatings = ratingsFileDF            //movieRatings -> |MovieId|avg_rating|rating_count|
    .select(col("MovieId"), col("Rating"))
    .groupBy("MovieId")
    .agg(
      avg("Rating").as("avg_rating"),
      count("Rating").as("rating_count")
    )

  val movieRelevance = genomeScoresDF         //movieRelevance -> |MovieId|avg_relevance|
    .select(col("MovieId"), col("Relevance"))
    .groupBy("MovieId")
    .agg(avg("relevance").as("avg_relevance"))

  //combine the previous dataframes to get our complete dataframe
  val movieStats = movieRatings.join(movieRelevance, "MovieId")   //movieStats -> |MovieId|avg_rating|rating_count|avg_relevance|

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
  //here we will get a column in return with true/false statements regarding the comparisons.
  //we will use this column to perform a left_anti join to exclude the dominated movies
  //(we are not interested in performing a plain left since we will get nulls in return)
  val skylineDF = statsA.join(statsB, dominationCondition, "left_anti")

  //skylineDF.show()
  skylineDF
    .write
    .mode("overwrite")
    .option("header", true)
    .csv(outputPathQuery6)

  // End of Query 6
  // ==============================================================

  //Query 7: Correlation Between Tag Relevance and Average Ratings
  //Files Used: ratings.csv, genome-scores.csv
  //Objective: Estimate the Pearson correlation between movies average genome tag relevance,
  //and average user rating, across movies.

  val avgRatingPerMovieDF = ratingsFileDF               //ratingsFileDF -> |UserId|MovieId|Rating|
    .groupBy(col("MovieId"))
    .agg(avg("Rating").alias("avg_rating_per_movie"))   //avgRatingPerMovieDF -> |MovieId|avg_rating_per_movie|

  val avgTagRelevancePerMovie = genomeScoresDF                      //genomeScoresDF -> |movieId|tagId|relevance|
    .groupBy(col("MovieId"))
    .agg(avg("Relevance").alias("avg_tag_relevance_per_movie"))     //avgTagRelevancePerMovie -> |MovieId|avg_tag_relevance_per_movie|

  val joinedResult = avgRatingPerMovieDF
    .join(avgTagRelevancePerMovie, Seq("MovieId"), "inner")         //joinedResult -> |MovieId|avg_rating_per_movie|avg_tag_relevance_per_movie|

  val correlation = joinedResult.stat.corr("avg_rating_per_movie", "avg_tag_relevance_per_movie")

  //println(s"Pearson correlation: $correlation")
  //joinedResult.show()


  if (hdfs.exists(new org.apache.hadoop.fs.Path(outputPathQuery7 + "/Pearson"))) {
    hdfs.delete(new org.apache.hadoop.fs.Path(outputPathQuery7 + "/Pearson"), true)
  }
  spark.sparkContext.parallelize(Seq(s"Pearson correlation: $correlation"))
    .coalesce(1)
    .saveAsTextFile(outputPathQuery7 + "/Pearson")

  joinedResult
    .write
    .mode("overwrite")
    .option("header", true)
    .csv(outputPathQuery7 + "/Averages")

  // End of Query 7
  // ==============================================================

  
  //** SWAPED QUERIES 8 AND 9 SINCE QUERY 8 IS MORE TIME CONSUMING.
  //Query 9: Tag-Relevance Anomaly — Overhyped Low-Rated Movies
  //Files Used: genome-scores.csv, ratings.csv, genome-tags.csv
  //Objective: Identify movies that are highly relevant (>=0.8) to popular tags of this list:
  //(“action”,“classic”, “thriller”), but have very low average user ratings <2.5.
  /* LOOSEN UP THE THRESHOLDS TO GET ANSWERS*/
  val famous_tags = genomeTagsDF //we filter only the tagIds for the desired tags
    .filter(col("Tag").isin("action", "classic", "thriller"))

  val overhypedMovies = famous_tags.join(genomeScoresDF, "TagId")  //we join the genome scores table to the filtered tags
    .groupBy("MovieId")                                            //then calculate the average relevance of each movie
    .agg(avg("Relevance").as("avg_relevance_to_tags"))
    .filter(col("avg_relevance_to_tags")>= 0.8 )
    .join(avgRatingPerMovieDF, "MovieId")                          //join the average ratings table
    .filter(col("avg_rating_per_movie") < 2.5)                     //filter those with average rating lower than 2.5

  //overhypedMovies.show()
  overhypedMovies
    .write
    .mode("overwrite")
    .option("header", true)
    .csv(outputPathQuery9)

  // End of Query 9
  // ==============================================================

  //Query 8: Reverse Nearest Neighbor — Match Users to a Movie’s Tag Vector
  //Files Used: ratings.csv, genome-scores.csv
  //Objective: Match users to a movie of your choice by comparing the average tag preferences of
  //their liked movies (rated with >4.0), with the tag profile of the target movie, using Cosine
  //Similarity

  val chosenMovieID = "79132" //Inception

  val tagRelevanceOfChosenMovie = genomeScoresDF   //tagRelevanceOfChosenMovie -> |TagId|Relevance| where MovieId == chosenMovieID
    .select(col("TagId"), col("Relevance"))
    .where(col("MovieId") === chosenMovieID)       // This is the tag profile of the chosen movie

  val likedMoviesDF = ratingsFileDF         //likedMoviesDF -> |UserId|MovieId| of movies rated by user > 4.0 (user's liked movies)
    .select(col("UserId"), col("MovieId"))
    .where(col("Rating") > 4.0)             //filter only the liked movies of each user (Rating > 4.0)

  val joinedByMovies = likedMoviesDF        //joinedByMovies -> |MovieId|UserId|TagId|Relevance|
    .join(genomeScoresDF, Seq("MovieId"))

  val avgTagRelevancePerUser = joinedByMovies     //avgTagRelevancePerUser -> |UserId|TagId|avg_tag_relevance_per_user|
    .repartition(400, col("UserId"))
    .groupBy(col("UserId"), col("TagId"))
    .agg(avg("Relevance").alias("avg_tag_relevance_per_user"))  // Compute the average of each tag relevance of all user's liked movies
    .repartition(400, col("TagId"))

  val tagProfilesJoined = avgTagRelevancePerUser    //tagProfilesJoined -> |UserId|TagId|user_score|target_score|
    .join(tagRelevanceOfChosenMovie, Seq("TagId"))
    .select(
      col("UserId"),
      col("TagId"),
      col("avg_tag_relevance_per_user").alias("user_score"),    //Some renaming
      col("Relevance").alias("target_score")                    //Some renaming
    )
    .repartition(400, col("UserId"))

  val cosineComponentsDF = tagProfilesJoined
    .withColumn("dot", col("user_score") * col("target_score"))                   //Create a new column where product between the user score and target score is calculated
    .withColumn("user_norm_sqr", col("user_score") * col("user_score"))           //Create a new column where the square of user score is calculated
    .withColumn("target_norm_sqr", col("target_score") * col("target_score"))     //Create a new column where the square of target score is calculated
    .groupBy(col("UserId"))                                                       //Group By UserId
    .agg(                                                                         //Sum each column with itself
      sum("dot").alias("dot_product"),
      sum("user_norm_sqr").alias("user_norm_sqr"),
      sum("target_norm_sqr").alias("target_norm_sqr")
    )
    .withColumn("cosine_similarity",                                              //Compute the Cosine Similarity
      col("dot_product") / (sqrt(col("user_norm_sqr")) * sqrt(col("target_norm_sqr")))
    )                                                                             //Until now ->|UserId|dot_product|user_norm_sqr|target_norm_sqr|cosine_similarity|
    .select(col("UserId"),col("cosine_similarity"))                               //cosineComponentsDF -> |UserId|cosine_similarity|
    .persist()

  //cosineComponentsDF.show()
  cosineComponentsDF
    .write
    .mode("overwrite")
    .option("header", true)
    .csv(outputPathQuery8)

  // End of Query 8
  // ==============================================================

  //Query 10: Reverse Top-K Neighborhood Users Using Semantic Tag Profiles
  //Files Used: ratings.csv, genome-scores.csv
  //Objective: Given a target movie, find the top-K users whose semantic tag profiles (i.e., tag
  //preferences inferred from high-rated (>4.0) movies) are closest (based on Cosine Similarity) to
  //that movie’s tag profile.

  val topK = 10
  val topKUsers = cosineComponentsDF
    .orderBy(col("cosine_similarity").desc)
    .limit(topK)
  val topKRanked = topKUsers.withColumn("Rank", row_number().over(Window.orderBy(col("cosine_similarity").desc))) //adding rank column

  topKRanked
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", true)
    .csv(outputPathQuery10)

  cosineComponentsDF.unpersist()    //maybe unecessary since it is right before the end of program, but good practice nontheless
  // End of Query 10
  // ==============================================================

  spark.stop()
}
