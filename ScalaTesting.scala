import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{avg, col, count, row_number, sqrt, sum}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}

object Main extends App{
  val spark = SparkSession.builder
    .master("local[*]")
    .appName("Scala Project")
    .getOrCreate()

  //Defining the Paths of the data
  val moviesPathHDFS = "hdfs://localhost:9000/ml-latest/movies.csv"
  val ratingsPathHDFS = "hdfs://localhost:9000/ml-latest/ratings.csv"
  val tagsPathHDFS = "hdfs://localhost:9000/ml-latest/tags.csv"
  val linksPathHDFS = "hdfs://localhost:9000/ml-latest/links.csv"
  val genomeTagsPathHDFS = "hdfs://localhost:9000/ml-latest/genome-tags.csv"
  val genomeScoresPathHDFS = "hdfs://localhost:9000/ml-latest/genome-scores.csv"

  //FOR DEBUGGING ONLY PURPOSES
  //val genomeScoresPathHDFS = "hdfs://localhost:9000/ml-small/genome-scores-trimmed.csv"
  //val ratingsPathHDFS = "hdfs://localhost:9000/ml-small/ratings_trimmed.csv"

  //Defining the Output path of the query answers
  val outputPath = "hdfs://localhost:9000/ml-latest/"

  val sc = spark.sparkContext

  //Defining Schemas
  val ratingsSchema = StructType(Seq(
    StructField("UserId", StringType, nullable = false),
    StructField("MovieId", StringType, nullable = false),
    StructField("Rating", DoubleType, nullable = false),
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
  val rawRatingsRDD = sc.textFile(ratingsPathHDFS)
  val ratingsHeader = rawRatingsRDD.first()

  val ratingsRDD = rawRatingsRDD
    .filter(row => row != ratingsHeader)                        //skip the csv headers.
    .map(line => line.split(","))                               //break the string into its components
    .map(fields => (fields(0), fields(1), fields(2).toDouble))  //(userId, movieId, rating)

  //tags.csv
  //We need to make the rdd this way for this file because some tags have commas
  val tagsRDD = tagsFileDF
    .select("UserId", "MovieId", "Tag")
    .rdd
    .map(row => (row.getAs[String]("UserId"), row.getAs[String]("MovieId"), row.getAs[String]("Tag"))) //(userId, movieId, tag)

  //genome-scores.csv
  val rawGenomeScoresRDD = sc.textFile(genomeScoresPathHDFS)
  val genomeScoresHeader = rawGenomeScoresRDD.first()

  val genomeScoresRDD = rawGenomeScoresRDD
    .filter(row => row != genomeScoresHeader)                   //skip the csv headers.
    .map(line => line.split(",",3))                               //break the string into its components
    .map(fields => (fields(0), fields(1), fields(2).toDouble))           //(movieId, tagId, relevance)

  //genome-tags.csv
  val rawGenomeTagsRDD = sc.textFile(genomeTagsPathHDFS)
  val genomeTagsHeader = rawGenomeTagsRDD.first()

  val genomeTagsRDD = rawGenomeTagsRDD
    .filter(row => row != genomeTagsHeader)          //skip the csv headers.
    .map(line => line.split(",",2))                  //break the string into its components
    .map(fields => (fields(0), fields(1)))           //(tagId,tag)

  //Query 1: Iceberg Query — Top Tags by Genre with High Ratings
  //Files used: movies.csv, ratings.csv, tags.csv
  //Objective: Find genre-tag pairs that appear in more than 100 movies
  //           and have an average user rating greater than 4.0.

  //Finding the average Rating per movie
  val avgRatingPerMovie = ratingsRDD                                    // avgRatingPerMovie -> (movieId, avg(Rating))
    .map { case (userId, movieId, rating) => (movieId, (rating, 1)) }   // (movieId,(rating, 1))
    .reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))                  // sum all the ratings and all the counts respectively
    .mapValues { case (sum, count) => sum / count }                     // compute the average (movieId, average)

  // Getting movieId - tag pairs (dropping the userId)
  val movieTagPairs = tagsRDD                                 //movieTagPairs -> (movieId,tag)
    .map { case (userId, movieId, tag) => (movieId, tag) }

  // Getting movieId - Genre pairs
  val movieGenrePairs = moviesRDD                             //movieGenrePairs -> (movieId,genre)
    .flatMap { case (movieId, title, genres) =>
      genres
        .filter(genre => genre != "(no genres listed)")
        .map( genre => (movieId, genre))
    }

  //Join movieId - tag pairs with movieId - Genre pairs on movieId
  val tagGenreJoinedWithMovies = movieTagPairs       //tagGenreJoinedWithMovies -> (movieId, (Tag,Genre))
    .join(movieGenrePairs)
    //.distinct()

  val movieByTagGenreRating = tagGenreJoinedWithMovies     //movieByTagGenreRating -> (movieId ,((Tag,Genre),avgMovieRating))
    .join(avgRatingPerMovie)

  val tagGenreAndRatings = movieByTagGenreRating                                  //tagGenreAndRatings -> (Tag,Genre),List(avg_rating_of_each_movie_associated)
    .map{ case(movieId, ((tag,genre),avgRating)) => ((tag,genre), avgRating)}     //((Tag,Genre),avgMovieRating)
    .groupByKey()                                                                 //group by tag-genre pair

  val filteredGroupByTagGenrePair = tagGenreAndRatings            //filteredGroupByTagGenrePair -> filtered((Tag,Genre),List(avg_rating_of_each_movie_associated))
    .filter(_._2/*.toSet*/.size > 100)                            //filter out the tag-genre pairs that have less than 100 average movie ratings associated with them(and thus movies)


  val icebergResults = filteredGroupByTagGenrePair
    .mapValues(list => {
      list
        .map( rating => (rating,1))
        .reduce((x,y) => (x._1 + y._1, x._2 + y._2))        //Compute the sum and count of the rating associated with each pair
    })
    .mapValues(tuple => tuple._1 / tuple._2)                //compute average
    .filter(tuple => tuple._2 > 4.0)                        //filter out the pairs that  have an average lower than 4.0


  icebergResults.saveAsTextFile("hdfs://localhost:9000/ml-latest/query1_output")
  //icebergResults.foreach(println)

  // End of Query 1
  // ==============================================================

  //Query 2: Tag Dominance per Genre — Most Used Tags with Ratings
  //Files Used: movies.csv, ratings.csv, tags.csv
  //Objective: For each movie genre, find the most commonly used tag,
  //           and return the average rating of the movies it was used on.

  val genreTagCounts = tagGenreJoinedWithMovies           //We get the (movieId,(Tag,Genre))
    .map { case(movieId,(tag,genre)) =>
      ((genre,tag), 1)                                    //we transform them to ((Genre,Tag), 1)
      }
    .reduceByKey(_ + _)                                   //we count the instances of each pair


  val mostUsedTagPerGenre = genreTagCounts                //mostUsedTagPerGenre -> (genre, (tag, count)), count = max, tag = most popular
    .map{ case((genre,tag),count) =>
      (genre, (tag, count))                               // Make genre the key
    }
    .reduceByKey { (a, b) => if (a._2 >= b._2) a else b } // Pick tag with the highest count for each genre

  val tagGenrePairsByMovie  = tagGenreJoinedWithMovies
    .map { case (movieId, (tag, genre)) =>               //we transform from (movieId, (tag, genre))
      ((genre,tag), movieId)                             //to ((genre,tag), movieId)
  }

  val dominantTagGenre = mostUsedTagPerGenre
    .map {
      case (genre, (tag, count)) => ((genre, tag), count) //we make the key to be (genre,tag) so we can join with tagGenrePairsByMovie
    }

  val filteredMovieTagGenre = tagGenrePairsByMovie
    .join(dominantTagGenre)                                               // ((genre, tag), (movieId, count))
    .map { case ((genre, tag), (movieId, _)) => (movieId, (genre, tag)) } //we now have the movies of each pair

  val movieGenreRatings = filteredMovieTagGenre
    .join(avgRatingPerMovie)  // (movieId, ((genre, tag), avgRating))     //get the avg rating of each movie

  val genreRatingPairs = movieGenreRatings
    .map { case ( _, ( (tag,genre), rating)) =>
      ((genre,tag), (rating, 1))                                // Convert to ((genre,tag), (rating, 1)) for averaging
    }

  val avgRatingPerGenreTag = genreRatingPairs
    .reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2))          // Sum ratings and counts
    .mapValues(x => x._1 / x._2)                                // Compute average rating per genre

  avgRatingPerGenreTag.saveAsTextFile("hdfs://localhost:9000/ml-latest/query2_output")
  //avgRatingPerGenreTag.foreach(println)

  // End of Query 2
  // ==============================================================

  //Query 3: Iceberg — Tags That Are Both Popular and Relevant
  //Files Used: genome-scores.csv, genome-tags.csv
  //Objective: Find tags that appear in over 100 unique movies and have an average relevance > 0.8.
  val genomeScoresGrouped = genomeScoresRDD
    .map(tuple => (tuple._2,(tuple._1,tuple._3))) //(tagId,(movieId,relevance))
    .groupByKey()                                 //(tagId,list(movieId,relevance))

  val filteredTags = genomeScoresGrouped
    .mapValues{iterable => iterable               //go inside the list and for each (movieId,relevance) element,
      .filter(_._2 > 0.6)                         //filter out those with relevance < 0.6 (filtering around 0.4-0.6 is good)
    }
    .filter(_._2.map(_._1).toSet.size > 100)        //count the unique movies associated with each tag and filter out those that have < 100 movies
    .mapValues{list=> list                          //go inside the list again and for each (movieId,relevance) element,
      .map(tuple =>(tuple._2, 1))                   //keep only the tuple (relevance, 1)
      .reduce((x,y) => (x._1 + y._1, x._2 + y._2))  //sum the relevance and the counts
    }
    .mapValues{case (sum,count) => sum/count}       //compute the average relevance
    .filter(_._2 > 0.8)                             //filter out whole tuples where avg_relevance < 0.8. filteredTags -> (tagId, avg_relevance)


  val finalFilteredTags = genomeTagsRDD             //genomeTagsRDD -> (tagId,tag)
    .join(filteredTags)                             //filteredTags -> (tagId, avg_relevance), join by tagId
    .map(tuple => tuple._2)                         //finalFilteredTags -> (tag, avg_relevance)

  finalFilteredTags.saveAsTextFile("hdfs://localhost:9000/ml-latest/query3_output")
  //finalFilteredTags.foreach(println)

  // End of Query 3
  // ==============================================================

  //Query 4: Sentiment Estimation — Tags Associated with Ratings
  //Files Used: tags.csv, ratings.csv
  //Objective: Compute the average user rating for each user-assigned tag from tags.csv

  val tagByUserMovie = tagsRDD                            //tagsRDD ->(userId, movieId, tag)
    .map(t => ((t._1, t._2, t._3))) // (userId, movieId, tag)
    //.distinct()                   //we can put distinct here if we don't want to compute duplicate tags by same user(if same user spams the same tag multiple times)
    .map(tuple3 => ((tuple3._1, tuple3._2), tuple3._3))   //((userId, movieId), tag)

  val ratingByUserMovie = ratingsRDD                      //ratingsRDD -> (userId, movieId, rating)
    .map(tuple3 => ((tuple3._1, tuple3._2), tuple3._3))   //((userId, movieId), rating)

  val averageOfEachTag = tagByUserMovie
    .join(ratingByUserMovie)                              //( (userId, movieId),(tag, rating) )
    .map {case (_, (tag, rating)) => (tag, (rating, 1))}  //(tag, (rating, 1)), we don't need userId,movieId now
    .reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))    //compute sum of ratings and the count of those ratings associated with each tag
    .mapValues { case (total, count) => total / count}    //compute the average rating associated with each tag

  averageOfEachTag.saveAsTextFile("hdfs://localhost:9000/ml-latest/query4_output")
  //averageOfEachTag.foreach(println)

  // End of Query 4
  // ==============================================================


  //Query 5: Multi-Iceberg Skyline Over Genre-Tag-User Triads
  //Files Used: movies.csv, tags.csv, ratings.csv
  //Objective: For each (genre, tag) pair that appears in more than 200 movies, calculate:
  //the average rating, the number of unique users who applied the tag. Then, compute a skyline
  //over the remaining (genre, tag) including the (genre, tag) pairs, for which no other (genre, tag)
  //dominates it in both average rating and user count.

  val popularGenreTagPairsAndRatings = tagGenreAndRatings //tagGenreAndRatings -> ((Tag,Genre),List(avg_rating_of_each_movie_associated))
    .filter(_._2.toSet.size > 200)                        //filter out the tag-genre pair if it is associated with less than 200 unique movies
    .mapValues {list =>
      list
        .map(rating => (rating, 1))
        .reduce((x, y) => (x._1 + y._1, x._2 + y._2))     //Compute the sum and count of the rating associated with each pair
    }
    .map{case((tag,genre),tuple) => ((genre,tag),tuple._1 / tuple._2)}  //compute average, popularGenreTagPairsAndRatings-> ((genre,tag),avg_rating)

  val userTagByMovie = tagsRDD
    .map{ case(userId, movieId, tag) =>
      (movieId,(userId,tag))              //(movieId,(userId,tag))
    }

  val userByGenreTag = userTagByMovie               //userByGenreTag-> (genre,tag),userCount))
    .join(movieGenrePairs)                          //movieGenrePairs -> (movieId,genre)
    .map{ case (movieId, ((userId,tag),genre)) =>
      ((genre,tag), userId)                         //(genre,tag), userId)
    }
    .distinct()                                     //remove duplicate user (for same genre-tag pair)
    .groupByKey()                                   //group by genre-tag pair
    .mapValues(_.size)                              //count the userIds therefor the unique users

  val avgRatingAndUsersByGenreTag = popularGenreTagPairsAndRatings      //avgRatingAndUsersByGenreTag -> ((genre,tag),(avg_rating,userCount))
    .join(userByGenreTag)                                               //userByGenreTag-> (genre,tag),userCount))

  //avgRatingAndUsersByGenreTag.foreach(println)

  // we will perform cartesian() and get the dot product
  // basically get all possible pairs and then perform a case to compare
  // all entries of the output.
  // THIS IS THE MOST INTENSIVE QUERY SO FAR !!!

  val skylineRDD = avgRatingAndUsersByGenreTag
    .cartesian(avgRatingAndUsersByGenreTag)
    // The Dot product looks like this:
    // (Genre, Tag)  (Rating, Users) (Genre, Tag)  (Rating, Users)
    //  pair A       stats of A      pair A         stats of A
    //  pair A       stats of A      pair B         stats of B
    //  pair A       stats of A      pair C         stats of C
    //  pair A       stats of A      pair D         stats of D
    //  pair B       stats of B      pair A         stats of A
    //  ...
    .filter { case (a, b) => a._1 != b._1 }     //we remove the duplicates.
    .filter { case ((_, (ratingA, userA)), ((_, (ratingB, userB)))) =>
      //we apply the domination condition
      (ratingB >= ratingA && userB >= userA) && (ratingB > ratingA || userB > userA)
    }
    //at this point we only have the pairs where the B entry dominates the A entry
    //Now we only keep the genre-tag pairs that are dominated
    .map { case (a, _) => (a._1, a._2) }
    //however there might be a chance that multiple entries on part B of the cartesian
    //dominate the same entry on part A. Therefore we need to use distinct to get rid of
    //excess copies of the same genre-tag pair.
    .distinct()

  //finally, from our avgRatingAndUsersByGenreTag
  //we want to keep the genre-tag pairs that don't belong in the dominated Keys.
  val finalSkylineRDD = avgRatingAndUsersByGenreTag.subtractByKey(skylineRDD)

  finalSkylineRDD.saveAsTextFile("hdfs://localhost:9000/ml-latest/query5_output")
  //finalSkylineRDD.foreach(println)

  // End of Query 5
  // ==============================================================

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
    .option("header", true)
    .csv("hdfs://localhost:9000/ml-latest/query6_output")

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

  spark.sparkContext.parallelize(Seq(s"Pearson correlation: $correlation"))
    .saveAsTextFile("hdfs://localhost:9000/ml-latest/query7_output_Pearson")

  joinedResult
    .write
    .option("header", true)
    .csv("hdfs://localhost:9000/ml-latest/query7_output_Averages")

  // End of Query 7
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
    .groupBy(col("UserId"), col("TagId"))
    .agg(avg("Relevance").alias("avg_tag_relevance_per_user"))  // Compute the average of each tag relevance of all user's liked movies

  val tagProfilesJoined = avgTagRelevancePerUser    //tagProfilesJoined -> |UserId|TagId|user_score|target_score|
    .join(tagRelevanceOfChosenMovie, Seq("TagId"))
    .select(
      col("UserId"),
      col("TagId"),
      col("avg_tag_relevance_per_user").alias("user_score"),    //Some renaming
      col("Relevance").alias("target_score")                    //Some renaming
    )

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

  //cosineComponentsDF.show()
  cosineComponentsDF
    .write
    .option("header", true)
    .csv("hdfs://localhost:9000/ml-latest/query8_output")

  // End of Query 8
  // ==============================================================

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
    .option("header", true)
    .csv("hdfs://localhost:9000/ml-latest/query9_output")

  // End of Query 9
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
    .write
    .option("header", true)
    .csv("hdfs://localhost:9000/ml-latest/query10_output")

  // End of Query 10
  // ==============================================================

  spark.stop()
}
