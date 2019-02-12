/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

  val data = RDDGenerator.genRDD[(Int, Int)](sc)(Arbitrary.arbitrary[(Int, Int)])

  property("save as a text file") {
    forAll(data) { rdd =>
      val out = new File(tempDir, "op-richrdd-" + DateTime.now().getMillis).toString
      rdd.saveAsTextFile(out, None, new JobConf(rdd.context.hadoopConfiguration))
      spark.read.textFile(out).count() shouldBe rdd.count()
    }
  }
  property("save as a compressed text file") {
    forAll(data) { rdd =>
      val out = new File(tempDir, "op-richrdd-" + DateTime.now().getMillis).toString
      rdd.saveAsTextFile(out, Some(classOf[DefaultCodec]), new JobConf(rdd.context.hadoopConfiguration))
      spark.read.textFile(out).count() shouldBe rdd.count()
    }
  }

}
