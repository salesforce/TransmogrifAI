/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.test

import java.io.File

import com.salesforce.op.utils.kryo.OpKryoRegistrator
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.scalatest.Suite

/**
 * Trait to enable Spark context for tests
 * Modified version of Spark 2.x test suite trait
 */
trait TestSparkContext extends TempDirectoryTest with TestCommon {
  self: Suite =>

  // Remove Logging of OWLQN and LBFGS used in LogisticRegression
  Logger.getLogger("breeze.optimize.OWLQN").setLevel(Level.WARN)
  Logger.getLogger("breeze.optimize.LBFGS").setLevel(Level.WARN)

  lazy val kryoClasses: Array[Class[_]] = Array(
    classOf[com.salesforce.op.test.Passenger],
    classOf[com.salesforce.op.test.PassengerCSV]
  )

  lazy val conf: SparkConf = new SparkConf()
    .setMaster("local[2]")
    .setAppName("op-test")
    .registerKryoClasses(kryoClasses)
    .set("spark.serializer", classOf[org.apache.spark.serializer.KryoSerializer].getName)
    .set("spark.kryo.registrator", classOf[OpKryoRegistrator].getName)
    .set("spark.ui.enabled", false.toString)

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
