import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, avg, count}
import org.apache.spark.sql.types.{StringType, StructField, StructType, DoubleType}

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

/*
  //Query 1: Iceberg Query — Top Tags by Genre with High Ratings
  //Files used: movies.csv, ratings.csv, tags.csv
  //Objective: Find genre-tag pairs that appear in more than 100 movies
  //           and have an average user rating greater than 4.0.

  //Finding the average Rating per movie
  val avgRatingPerMovie = ratingsRDD
    .map { case (userId, movieId, rating) => (movieId, (rating, 1)) }   // movieId → (rating, 1)
    .reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))                  // sum all the ratings and all the counts respectively
    .mapValues { case (sum, count) => sum / count }                     // compute the average (movieId, average)

  // Getting movieId - tag pairs (dropping the userId)
  val movieTagPairs = tagsRDD
    .map { case (userId, movieId, tag) => (movieId, tag) }    //(movieId,tag)


  //WITH DISTINCT, QUERY 1 RETURNS NOTHING, WITHOUT DISTINCT, IT RETURNS SOME ANSWERS
  //ACTUALLY, NOT HAVING DISTINCT MAKES MORE SENSE
  //.distinct()                                               //different user might have used the same movieId-tag pair so we make sure there are no duplicates

  // Getting movieId - Genre pairs
  val movieGenrePairs = moviesRDD
    .flatMap { case (movieId, title, genres) =>
      genres
        .filter(genre => genre != "(no genres listed)")
        .map( genre => (movieId, genre))                      //(movieId,genre)
    }

  //Join movieId - tag pairs with movieId - Genre pairs on movieId
  val tagGenreJoinedWithMovies = movieTagPairs.join(movieGenrePairs) //(movieId, (Tag,Genre))

  val movieByTagGenreRating = tagGenreJoinedWithMovies
    .join(avgRatingPerMovie)                                                      //movieByTagGenreRating -> (movieId ,((Tag,Genre),avgMovieRating))

  val tagGenreAndRatings = movieByTagGenreRating
    .map{ case(movieId, ((tag,genre),avgRating)) => ((tag,genre), avgRating)}     //((Tag,Genre),avgMovieRating)
    .groupBy(_._1)                                                                //group by tag-genre pair [(Tag,Genre) → List(((Tag,Genre),avg_rating_of_each_movie_associated))]
    .mapValues(list => list.map(tuple => tuple._2))                               //Simplify the values so we have (Tag,Genre) → List(avg_rating_of_each_movie_associated)


  val filteredGroupByTagGenrePair = tagGenreAndRatings
    .filter(_._2.size > 100)                            //filter out the tag-genre pairs that have less than 100 average movie ratings associated with them(and thus movies)


  val icebergResults = filteredGroupByTagGenrePair
    .mapValues(list => {
      list
        .map( rating => (rating,1))
        .reduce((x,y) => (x._1 + y._1, x._2 + y._2))        //Compute the sum and count of the rating associated with each pair
    })
    .mapValues(tuple => tuple._1 / tuple._2)                //compute average
    .filter(tuple => tuple._2 > 4.0)                        //filter out the pairs that  have an average lower than 4.0


  //icebergResults.saveAsTextFile("hdfs://localhost:9000/ml-latest/query1_output")
  //icebergResults.foreach(println)


  // End of Query 1
  // ==============================================================

  //Query 2: Tag Dominance per Genre — Most Used Tags with Ratings
  //Files Used: movies.csv, ratings.csv, tags.csv
  //Objective: For each movie genre, find the most commonly used tag,
  //           and return the average rating of the movies it was used on.

  val genreToTags = tagGenreJoinedWithMovies    //We get the movieId → (Tag,Genre)
    .map(pairs => (pairs._2._2, pairs._2._1))   //we transform them to (Genre,Tag)
    .groupByKey()                               //and we group by genre(So we have Genre -> List(Tags))


  val mostPopularTagPerGenre = genreToTags
    .mapValues { tagList =>
      tagList
        .map(tag => (tag, 1))                                   // Convert each tag to (tag, 1)
        .groupBy { pair => pair._1 }                            // group by tag
        .map { case (tag, list) => (tag, list.map(_._2).sum) }  // Count occurrences of each tag
        .maxBy(pair => pair._2)                                 //take the most popular(the one with the most counts)
    }

  val broadcastPopularTags = sc.broadcast(                     // Broadcast most popular tag per genre
    mostPopularTagPerGenre
      .map { case (genre, (tag, _)) => (genre, tag) }          // Strip out count, keep (genre, tag)
      .collectAsMap()                                          // Collect to map for broadcasting
  )

  val filteredMovies = tagGenreJoinedWithMovies.filter { case (_, (tag, genre)) =>    //(movieId, (tag, genre))
    broadcastPopularTags.value.get(genre).contains(tag)                               // Keep only movie-tag pairs with dominant tag
  }

  val movieGenreRatings = filteredMovies.join(avgRatingPerMovie) // Join with average rating per movie → (movieId,( (tag,genre), avg_rating ) )

  val genreRatingPairs = movieGenreRatings.map { case ( _, ( (tag,genre), rating)) =>
    ((genre,tag), (rating, 1))                                       // Convert to ((genre,tag), (rating, 1)) for averaging
  }

  val avgRatingPerGenreTag = genreRatingPairs
    .reduceByKey((x, y) => (x._1 + y._1, x._2 + y._2))          // Sum ratings and counts
    .mapValues(x => x._1 / x._2)                                // Compute average rating per genre

  //avgRatingPerGenreTag.saveAsTextFile("hdfs://localhost:9000/ml-latest/query2_output")
  avgRatingPerGenreTag.foreach(println)

  //QUESTION 1:DOES IT MATTER HOW WE RETURN THIS? ((genre,tag), avg_rating) or (genre, tag, avg_rating) or (genre, (tag, avg_rating))
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
      .filter(_._2 > 0.6)                         //filter out those with relevance < 0.6
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

  //QUESTION 2: HOW IS THE RELEVANCE METRIC CALCULATED? IS IT JACCARD INDEX?
  //finalFilteredTags.saveAsTextFile("hdfs://localhost:9000/ml-latest/query3_output")
  //finalFilteredTags.foreach(println)

  // End of Query 3
  // ==============================================================


  //Query 4: Sentiment Estimation — Tags Associated with Ratings
  //Files Used: tags.csv, ratings.csv
  //Objective: Compute the average user rating for each user-assigned tag from tags.csv

  val tagByUserMovie = tagsRDD                            //tagsRDD ->(userId, movieId, tag)
    .map(tuple3 => ((tuple3._1, tuple3._2), tuple3._3))   //((userId, movieId), tag)

  val ratingByUserMovie = ratingsRDD                      //ratingsRDD -> (userId, movieId, rating)
    .map(tuple3 => ((tuple3._1, tuple3._2), tuple3._3))   //((userId, movieId), rating)

  val averageOfEachTag = tagByUserMovie
    .join(ratingByUserMovie)                              //( (userId, movieId),(tag, rating) )
    .map {case (_, (tag, rating)) => (tag, (rating, 1))}  //(tag, (rating, 1)), we don't need userId,movieId now
    .reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))    //compute sum of ratings and the count of those ratings associated with each tag
    .mapValues { case (total, count) => total / count}    //compute the average rating associated with each tag

  //averageOfEachTag.saveAsTextFile("hdfs://localhost:9000/ml-latest/query4_output")
  //averageOfEachTag.foreach(println)

  // End of Query 4
  // ==============================================================

  // QUESTION: ARE WE MEANT TO APPLY THE SKYLINE ON THE FILTER PAIRS (200 > UNIQUE MOVIES)
  // ALSO IS THERE A SIMPLER WAY TO DO THE SKYLINE THAN CARTESIAN?
  // OR THE REMAINING PAIRS THAT ARE IN <= 200 UNIQUE MOVIES?
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

  val userByGenreTag = userTagByMovie               //userTagGenreByMovie -> (movieId, ((userId,tag),genre))
    .join(movieGenrePairs)                          //movieGenrePairs -> (movieId,genre)
    .map{ case (movieId, ((userId,tag),genre)) =>
      ((genre,tag), userId)                         //(genre,tag), userId)
    }
    .distinct()                                     //remove duplicate user (for same genre-tag pair)
    .groupByKey()                                   //group by genre-tag pair
    .mapValues(_.size)                              //count the userIds therefor the unique users

  val avgRatingAndUsersByGenreTag = popularGenreTagPairsAndRatings
    .join(userByGenreTag)                                               //userByGenreTag-> (genre,tag),userCount))

  //avgRatingAndUsersByGenreTag.foreach(println)

  // we will stay in the RDD area and perform cartesian() and get the dot product
  // basically get all possible pairs and then perform a case to compare
  // all entries of the output.
  // THIS IS THE MOST INTENSIVE QUERY  !!!

  val skylineRDD = avgRatingAndUsersByGenreTag.cartesian(avgRatingAndUsersByGenreTag)

    // The Dot product looks like this:
    // (Genre, Tag)  (Rating, Users) (Genre, Tag)  (Rating, Users)
    //  pair A       stats of A      pair A         stats of A
    //  pair A       stats of A      pair B         stats of B
    //  pair A       stats of A      pair C         stats of C
    //  pair A       stats of A      pair D         stats of D
    //  pair B       stats of B      pair A         stats of A
    // ...

    .filter { case (a, b) => a._1 != b._1 }     //we remove the duplicates.
    .filter { case ((_, (ratingA, userA)), ((_, (ratingB, userB)))) =>
      //we apply the domination condition
      (ratingB >= ratingA && userB >= userA) && (ratingB > ratingA || userB > userA)
    }
    //at this point we only have the pairs where the B entry dominates the A entry

    //Now we only keep the genre-tag pairs that are dominated
    //getting rid of the ratings and user counts.
    .map { case (a, _) => a._1 }

    //however there might be a chance that multiple entries on part B of the cartesian
    //dominate the same entry on part A. Therefore we need to use distinct to get rid of
    //excess copies of the same genre-tag pair.
    .distinct()

  //finally, from our avgRatingAndUsersByGenreTag
  //we want to keep the genre-tag pairs that don't belong in the dominated Keys.
  val finalSkylineRDD = avgRatingAndUsersByGenreTag
    .filter { case (key, _) => !skylineRDD.collect().contains(key) }

  finalSkylineRDD.foreach(println)

  //finalSkylineRDD.saveAsTextFile("hdfs://localhost:9000/ml-latest/query5_output1")
  // End of Query 5
  // ==============================================================
*/
/*
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

  //Question: we are creating logical references and not deep physical copies, is that ok? is this functional?
  val statsA = movieStats.alias("a")
  val statsB = movieStats.alias("b")

  //
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
  //skylineDF.rdd.saveAsTextFile("hdfs://localhost:9000/ml-latest/query6_output")

  // End of Query 6
  // ==============================================================
*/
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

  // End of Query 7
  // ==============================================================

  //Query 8: Reverse Nearest Neighbor — Match Users to a Movie’s Tag Vector
  //Files Used: ratings.csv, genome-scores.csv
  //Objective: Match users to a movie of your choice by comparing the average tag preferences of
  //their liked movies (rated with >4.0), with the tag profile of the target movie, using Cosine
  //Similarity

  val chosenMovieID = "79132"
  val tagRelevance = genomeScoresDF
    .select(col("MovieId"), col("TagId"), col("Relevance"))
    .where(col("MovieId") === chosenMovieID)

  val filterRatings = ratingsFileDF         //filterRatings -> |UserId|MovieId| of movies rated by user > 4.0 (user's liked movies)
    .select(col("UserId"), col("MovieId"))
    .where(col("Rating") > 4.0)

  val joinedByMovies = filterRatings        //joinedByMovies -> |MovieId|UserId|TagId|Relevance|
    .join(genomeScoresDF, Seq("MovieId"))

  ///////////////CURSED///////////////////////
//  val avgTagRelevancePerUser = joinedByMovies
//    .groupBy(col("UserId"), col("tagId"))
//    .agg(avg("Relevance").alias("avg_tag_relevance_per_user"))

  joinedByMovies.show()
    //avgTagRelevancePerUser.show()
    //filterRatings.show()
    //tagRelevance.show()

  // End of Query 8
  // ==============================================================


  //Query 9: Tag-Relevance Anomaly — Overhyped Low-Rated Movies
  //Files Used: genome-scores.csv, ratings.csv, genome-tags.csv
  //Objective: Identify movies that are highly relevant (>=0.8) to popular tags of this list:
  //(“action”,“classic”, “thriller”), but have very low average user ratings <2.5.

  val famous_tags = genomeTagsDF //we filter only the tagIds for the desired tags
    .filter(col("Tag").isin("action", "classic", "thriller"))

  val overhypedMovies = famous_tags.join(genomeScoresDF, "TagId")  //we join the genome scores table to the filtered tags
    .groupBy("MovieId")                                            //then calculate the average relevance of each movie
    .agg(avg("Relevance").as("avg_relevance_to_tags"))
    .filter(col("avg_relevance_to_tags")>= 0.8 )
    .join(avgRatingPerMovieDF, "MovieId")                          //join the average ratings table
    .filter(col("avg_rating_per_movie") < 2.5)                     //filter those with average rating lower than 2.5

  //overhypedMovies.show()
  //overhypedMovies.rdd.saveAsTextFile("hdfs://localhost:9000/ml-latest/query9_output")
  //QUESTION:ME THN PANW LOGIKH DEN YPARXEI KAMIA TAINIA POU NA EINAI OVERHYPED

  // End of Query 9
  // ==============================================================

  //Query 10: Reverse Top-K Neighborhood Users Using Semantic Tag Profiles
  //Files Used: ratings.csv, genome-scores.csv
  //Objective: Given a target movie, find the top-K users whose semantic tag profiles (i.e., tag
  //preferences inferred from high-rated (>4.0) movies) are closest (based on Cosine Similarity) to
  //that movie’s tag profile.

  // End of Query 10
  // ==============================================================

  //QUESTION:is it better to use .mapValues + .reduceByKey over .groupByKey when doing aggregations?
  spark.stop()
}
