/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.spark

import java.io.File

import com.holdenkarau.spark.testing.RDDGenerator
import com.salesforce.op.test.TestSparkContext
import org.apache.hadoop.io.compress.DefaultCodec
import org.apache.hadoop.mapred.JobConf
import org.joda.time.DateTime
import org.junit.runner.RunWith
import org.scalacheck.Arbitrary
import org.scalatest.PropSpec
import org.scalatest.junit.JUnitRunner
import org.scalatest.prop.PropertyChecks


@RunWith(classOf[JUnitRunner])
class RichRDDTest extends PropSpec with PropertyChecks with TestSparkContext {
  import com.salesforce.op.utils.spark.RichRDD._

  val data = RDDGenerator.genRDD[(Int, Int)](spark.sparkContext)(Arbitrary.arbitrary[(Int, Int)])

  property("save as a text file") {
    forAll(data) { rdd =>
      val out = new File(tempDir + "/op-richrdd-" + DateTime.now().getMillis).toString
      rdd.saveAsTextFile(out, None, new JobConf(rdd.context.hadoopConfiguration))
      spark.read.textFile(out).count() shouldBe rdd.count()
    }
  }
  property("save as a compressed text file") {
    forAll(data) { rdd =>
      val out = new File(tempDir + "/op-richrdd-" + DateTime.now().getMillis).toString
      rdd.saveAsTextFile(out, Some(classOf[DefaultCodec]), new JobConf(rdd.context.hadoopConfiguration))
      spark.read.textFile(out).count() shouldBe rdd.count()
    }
  }

}
