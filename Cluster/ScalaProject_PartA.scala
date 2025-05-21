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

  val outputPathQuery1 = outputPath + "/ProjectOutputs/Query1"
  val outputPathQuery2 = outputPath + "/ProjectOutputs/Query2"
  val outputPathQuery3 = outputPath + "/ProjectOutputs/Query3"
  val outputPathQuery4 = outputPath + "/ProjectOutputs/Query4"
  val outputPathQuery5 = outputPath + "/ProjectOutputs/Query5"

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

  if (hdfs.exists(new org.apache.hadoop.fs.Path(outputPathQuery1))) {
    hdfs.delete(new org.apache.hadoop.fs.Path(outputPathQuery1), true)
  }
  icebergResults.saveAsTextFile(outputPathQuery1)
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

  if (hdfs.exists(new org.apache.hadoop.fs.Path(outputPathQuery2))) {
    hdfs.delete(new org.apache.hadoop.fs.Path(outputPathQuery2), true)
  }
  avgRatingPerGenreTag
  .coalesce(1)
  .saveAsTextFile(outputPathQuery2)
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

  if (hdfs.exists(new org.apache.hadoop.fs.Path(outputPathQuery3))) {
    hdfs.delete(new org.apache.hadoop.fs.Path(outputPathQuery3), true)
  }
  finalFilteredTags.saveAsTextFile(outputPathQuery3)
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

  if (hdfs.exists(new org.apache.hadoop.fs.Path(outputPathQuery4))) {
    hdfs.delete(new org.apache.hadoop.fs.Path(outputPathQuery4), true)
  }
  averageOfEachTag.saveAsTextFile(outputPathQuery4)
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
    .repartition(200)

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

  if (hdfs.exists(new org.apache.hadoop.fs.Path(outputPathQuery5))) {
    hdfs.delete(new org.apache.hadoop.fs.Path(outputPathQuery5), true)
  }
  finalSkylineRDD.saveAsTextFile(outputPathQuery5)
  //finalSkylineRDD.foreach(println)

  // End of Query 5
  // ==============================================================
  spark.stop()
}
