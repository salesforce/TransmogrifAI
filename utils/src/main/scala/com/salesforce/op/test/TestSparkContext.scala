// scalastyle:off header.matches
/*
 * Modifications: (c) 2017, Salesforce.com, Inc.
 *
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.salesforce.op.test

import java.io.File

import com.salesforce.op.utils.kryo.OpKryoRegistrator
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.Suite

/**
 * Trait to enable Spark context for tests
 * Modified version of Spark 2.x test suite trait
 */
trait TestSparkContext extends TempDirectoryTest with TestCommon {
  self: Suite =>

  // Remove Breeze logging noise
  Logger.getLogger("breeze.optimize").setLevel(Level.WARN)

  lazy val kryoClasses: Array[Class[_]] = Array(
    classOf[com.salesforce.op.test.Passenger],
    classOf[com.salesforce.op.test.PassengerCSV]
  )

  val sparkLogAppender: MemoryAppender = {
    val sparkAppender = new MemoryAppender()
    sparkAppender.setName("spark-appender")
    sparkAppender.setThreshold(Level.INFO)
    sparkAppender.setLayout(new org.apache.log4j.PatternLayout)
    Logger.getRootLogger.addAppender(sparkAppender)
    sparkAppender
  }

  lazy val conf: SparkConf = {
    val conf = new SparkConf()
    conf
      .setMaster("local[2]")
      .setAppName(conf.get("spark.app.name", "op-test"))
      .registerKryoClasses(kryoClasses)
      .set("spark.serializer", classOf[org.apache.spark.serializer.KryoSerializer].getName)
      .set("spark.kryo.registrator", classOf[OpKryoRegistrator].getName)
      .set("spark.ui.enabled", false.toString) // Disables Spark Application UI
    // .set("spark.kryo.registrationRequired", "true") // Enable to debug Kryo
    // .set("spark.kryo.unsafe", "true") // This might improve performance
  }

  implicit lazy val spark: SparkSession = SparkSession.builder.config(conf).getOrCreate()
  implicit lazy val sc: SparkContext = spark.sparkContext

  lazy val checkpointDir: String = createDirectory(tempDir.getCanonicalPath, "checkpoints").toString

  override def beforeAll: Unit = {
    super[TempDirectoryTest].beforeAll()
    sc.setCheckpointDir(checkpointDir)
  }

  override def afterAll: Unit = {
    try {
      deleteRecursively(new File(checkpointDir))
      SparkSession.clearActiveSession()
      spark.stop()
    } finally {
      super[TempDirectoryTest].afterAll()
    }
  }
}

/**
 * Trait to enable Spark streaming context for tests
 */
trait TestSparkStreamingContext extends TestSparkContext {
  self: Suite =>

  implicit lazy val streaming: StreamingContext = StreamingContext.getActiveOrCreate(() =>
    new StreamingContext(sc, Seconds(1))
  )

  override def afterAll: Unit = {
    streaming.stop(stopSparkContext = false)
    super[TestSparkContext].afterAll
  }

}
