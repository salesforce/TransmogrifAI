/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._


object DataGeneration {


  val spark = SparkSession
    .builder()
    .appName("Data Generator")
    .getOrCreate()

  import spark.implicits._

  val df = spark.read.format("com.databricks.spark.csv").option("inferSchema", "true").load("PassengerData.csv")
  // /Users/lmcguire/Stuff/Projects/optimus-prime/test-data/resources/PassengerData.csv


  val names = Seq("passengerId", "age", "gender", "height", "weight",
    "description", "boarded", "recordDate", "survived")

  val df2 = df.toDF(names: _*)

  case class PassengerDG(passengerId: Int, age: Option[Int], gender: String,
                       height: Int, weight: Int, description: Option[String],
                       boarded: Long, recordDate: Long, survived: Boolean)

  val ds = df2.as[PassengerDG]

  df2.select("passengerId", "age", "gender", "height", "weight", "survived", "boarded").collect()

  val stringMap = udf { (s: String) => Map(s -> "string") }
  val boolMap = udf { (s: String) => Map(s -> false) }
  val numMap = udf { (s: String) => Map(s -> 1.0) }
  val passId = udf { (i: Int) => if (i > 6) 4 else i}

  val makeVector = udf { (s: String) =>
    if (s == "Female") Vectors.dense(1.0, 0.0)
    else Vectors.dense(0.0, 1.0)
  }

  val df3 = df2
    .withColumn("stringMap", stringMap(col("gender")))
    .withColumn("numericMap", numMap(col("gender")))
    .withColumn("booleanMap", boolMap(col("gender")))

  val df4 = df3.select("passengerId", "age", "gender", "height", "weight", "description",
    "survived", "boarded", "recordDate", "stringMap", "numericMap", "booleanMap")

  df4.collect()

  df4.write
    .options(Map("recordName" -> "Passenger", "recordNamespace" -> "com.salesforce.op.test"))
    .avro("Passenger.avro")

}
